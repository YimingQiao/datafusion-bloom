use datafusion::arrow::array::{
    Array, ArrayRef, BooleanArray, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use datafusion::arrow::buffer::BooleanBuffer;
use datafusion::common::Result;
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};

/// A compact membership structure produced by transfer.
///
/// Most instances are temporary; a terminal Direct handoff may retain one in
/// the formal scan. Its storage therefore remains registered with DataFusion's
/// memory pool for exactly as long as the structure is reachable.
#[derive(Debug)]
pub(crate) struct TransferBloomFilter {
    storage: FilterStorage,
    _reservation: Option<MemoryReservation>,
}

#[derive(Debug)]
enum FilterStorage {
    Bloom {
        words: Box<[u64]>,
        bit_mask: usize,
        probes: u32,
        is_empty: bool,
    },
    DenseInteger {
        words: Box<[u64]>,
        minimum: i128,
        maximum: i128,
    },
}

impl TransferBloomFilter {
    #[cfg(test)]
    pub(crate) fn with_capacity(expected_items: usize, false_positive_rate: f64) -> Self {
        let (bit_count, probes) = bloom_shape(expected_items, false_positive_rate);
        Self {
            storage: FilterStorage::Bloom {
                words: vec![0; bit_count / 64].into_boxed_slice(),
                bit_mask: bit_count - 1,
                probes,
                is_empty: true,
            },
            _reservation: None,
        }
    }

    pub(crate) fn try_with_capacity(
        expected_items: usize,
        false_positive_rate: f64,
        context: &TaskContext,
    ) -> Result<Self> {
        let (bit_count, probes) = bloom_shape(expected_items, false_positive_rate);
        let reservation = reserve_filter_bytes(bit_count.div_ceil(8), context)?;
        Ok(Self {
            storage: FilterStorage::Bloom {
                words: vec![0; bit_count / 64].into_boxed_slice(),
                bit_mask: bit_count - 1,
                probes,
                is_empty: true,
            },
            _reservation: Some(reservation),
        })
    }

    /// Build an exact bitmap when an integer domain is compact enough.
    #[cfg(test)]
    pub(crate) fn dense_integer(minimum: i128, maximum: i128, max_bits: usize) -> Option<Self> {
        let span = maximum.checked_sub(minimum)?.checked_add(1)?;
        let bit_count = usize::try_from(span).ok()?;
        if bit_count == 0 || bit_count > max_bits {
            return None;
        }
        Some(Self {
            storage: FilterStorage::DenseInteger {
                words: vec![0; bit_count.div_ceil(64)].into_boxed_slice(),
                minimum,
                maximum,
            },
            _reservation: None,
        })
    }

    pub(crate) fn try_dense_integer(
        minimum: i128,
        maximum: i128,
        max_bits: usize,
        context: &TaskContext,
    ) -> Result<Option<Self>> {
        let Some(span) = maximum
            .checked_sub(minimum)
            .and_then(|span| span.checked_add(1))
        else {
            return Ok(None);
        };
        let Ok(bit_count) = usize::try_from(span) else {
            return Ok(None);
        };
        if bit_count == 0 || bit_count > max_bits {
            return Ok(None);
        }
        let word_count = bit_count.div_ceil(64);
        let reservation = reserve_filter_bytes(word_count.saturating_mul(8), context)?;
        Ok(Some(Self {
            storage: FilterStorage::DenseInteger {
                words: vec![0; word_count].into_boxed_slice(),
                minimum,
                maximum,
            },
            _reservation: Some(reservation),
        }))
    }

    pub(crate) fn is_dense_integer(&self) -> bool {
        matches!(self.storage, FilterStorage::DenseInteger { .. })
    }

    /// Inclusive bounds of an exact integer bitmap. These bounds are a sound,
    /// redundant Parquet pruning hint; membership remains the semantic row
    /// predicate regardless of whether it runs inside or above the reader.
    pub(crate) fn integer_bounds(&self) -> Option<(i128, i128)> {
        match &self.storage {
            FilterStorage::DenseInteger {
                minimum, maximum, ..
            } => Some((*minimum, *maximum)),
            FilterStorage::Bloom { .. } => None,
        }
    }

    #[inline]
    pub(crate) fn insert_integer(&mut self, value: i128) {
        let FilterStorage::DenseInteger {
            words,
            minimum,
            maximum,
        } = &mut self.storage
        else {
            return;
        };
        if value < *minimum || value > *maximum {
            return;
        }
        let bit = (value - *minimum) as usize;
        words[bit >> 6] |= 1_u64 << (bit & 63);
    }

    /// Evaluate an exact integer bitmap without hashing. `None` means the
    /// array type is not an integer type supported by the bitmap.
    pub(crate) fn integer_mask(&self, array: &ArrayRef) -> Option<BooleanArray> {
        let FilterStorage::DenseInteger {
            words,
            minimum,
            maximum,
            ..
        } = &self.storage
        else {
            return None;
        };
        macro_rules! mask {
            ($array_type:ty) => {{
                let values = array.as_any().downcast_ref::<$array_type>()?;
                let mask = BooleanBuffer::collect_bool(values.len(), |index| {
                    if !values.is_valid(index) {
                        return false;
                    }
                    let value = values.value(index) as i128;
                    if value < *minimum || value > *maximum {
                        return false;
                    }
                    let bit = (value - *minimum) as usize;
                    words[bit >> 6] & (1_u64 << (bit & 63)) != 0
                });
                Some(BooleanArray::new(mask, None))
            }};
        }
        match array.data_type() {
            datafusion::arrow::datatypes::DataType::Int8 => mask!(Int8Array),
            datafusion::arrow::datatypes::DataType::Int16 => mask!(Int16Array),
            datafusion::arrow::datatypes::DataType::Int32 => mask!(Int32Array),
            datafusion::arrow::datatypes::DataType::Int64 => mask!(Int64Array),
            datafusion::arrow::datatypes::DataType::UInt8 => mask!(UInt8Array),
            datafusion::arrow::datatypes::DataType::UInt16 => mask!(UInt16Array),
            datafusion::arrow::datatypes::DataType::UInt32 => mask!(UInt32Array),
            datafusion::arrow::datatypes::DataType::UInt64 => mask!(UInt64Array),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn insert(&mut self, hash: u64) {
        let FilterStorage::Bloom {
            words,
            bit_mask,
            probes,
            is_empty,
        } = &mut self.storage
        else {
            return;
        };
        *is_empty = false;
        let (first, step) = split_hash(hash);
        for probe in 0..*probes {
            let bit = first.wrapping_add((probe as u64).wrapping_mul(step)) as usize & *bit_mask;
            words[bit >> 6] |= 1_u64 << (bit & 63);
        }
    }

    #[inline]
    pub(crate) fn might_contain(&self, hash: u64) -> bool {
        let FilterStorage::Bloom {
            words,
            bit_mask,
            probes,
            is_empty,
        } = &self.storage
        else {
            return false;
        };
        if *is_empty {
            return false;
        }

        let (first, step) = split_hash(hash);
        for probe in 0..*probes {
            let bit = first.wrapping_add((probe as u64).wrapping_mul(step)) as usize & *bit_mask;
            if words[bit >> 6] & (1_u64 << (bit & 63)) == 0 {
                return false;
            }
        }
        true
    }

    #[cfg(test)]
    pub(crate) fn bit_count(&self) -> usize {
        match &self.storage {
            FilterStorage::Bloom { words, .. } | FilterStorage::DenseInteger { words, .. } => {
                words.len() * 64
            }
        }
    }
}

fn bloom_shape(expected_items: usize, false_positive_rate: f64) -> (usize, u32) {
    let item_count = expected_items.max(1);
    let ln_2 = std::f64::consts::LN_2;
    let ideal_bits = -((item_count as f64) * false_positive_rate.ln()) / (ln_2 * ln_2);
    let bit_count = (ideal_bits.ceil() as usize).max(64).next_power_of_two();
    let probes = (((bit_count as f64 / item_count as f64) * ln_2).round() as u32).clamp(1, 16);
    (bit_count, probes)
}

fn reserve_filter_bytes(bytes: usize, context: &TaskContext) -> Result<MemoryReservation> {
    let reservation =
        MemoryConsumer::new("BloomTransferMembership").register(context.memory_pool());
    reservation.try_grow(bytes)?;
    Ok(reservation)
}

#[inline]
fn split_hash(hash: u64) -> (u64, u64) {
    let first = avalanche(hash ^ 0x9e37_79b9_7f4a_7c15);
    // An odd step visits every position when the number of bits is a power of two.
    let step = avalanche(hash.rotate_left(31) ^ 0xd6e8_feb8_6659_fd93) | 1;
    (first, step)
}

#[inline]
fn avalanche(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{ArrayRef, BooleanArray, Int32Array};

    use super::TransferBloomFilter;

    #[test]
    fn inserted_hashes_are_never_rejected() {
        let mut filter = TransferBloomFilter::with_capacity(10_000, 0.01);
        for value in 0..10_000_u64 {
            filter.insert(value.wrapping_mul(17));
        }
        for value in 0..10_000_u64 {
            assert!(filter.might_contain(value.wrapping_mul(17)));
        }
        assert!(filter.bit_count().is_power_of_two());
    }

    #[test]
    fn an_empty_filter_rejects_everything() {
        let filter = TransferBloomFilter::with_capacity(0, 0.01);
        assert!(!filter.might_contain(0));
        assert!(!filter.might_contain(u64::MAX));
    }

    #[test]
    fn dense_integer_filter_is_exact_and_null_safe() {
        let mut filter = TransferBloomFilter::dense_integer(10, 20, 64).unwrap();
        filter.insert_integer(10);
        filter.insert_integer(17);
        filter.insert_integer(20);
        let values = Arc::new(Int32Array::from(vec![
            Some(9),
            Some(10),
            None,
            Some(17),
            Some(18),
            Some(20),
            Some(21),
        ])) as ArrayRef;
        assert_eq!(
            filter.integer_mask(&values).unwrap(),
            BooleanArray::from(vec![false, true, false, true, false, true, false])
        );
    }
}
