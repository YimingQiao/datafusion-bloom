//! Evaluate transfer membership over samples and materialized handoffs.

use super::*;

/// Evaluate transfer membership for an estimate without mutating or
/// re-owning the sampled/materialized batches.
pub(super) fn count_survivors(
    partitions: &[Vec<RecordBatch>],
    filters: &[CascadeFilter],
    random_state: &RandomState,
) -> Result<usize> {
    let mut total = 0;
    for batch in partitions.iter().flatten() {
        let mut retained = vec![true; batch.num_rows()];
        for cascade in filters {
            let mask = evaluate_membership(batch, &cascade.keys, &cascade.filter, random_state)?;
            for (keep, member) in retained.iter_mut().zip(mask.values()) {
                *keep &= member;
            }
        }
        total += retained.into_iter().filter(|keep| *keep).count();
    }
    Ok(total)
}

/// Physically apply newly arrived transfer restrictions and immediately reset
/// batch ownership, preventing filtered views from retaining their old input.
pub(super) fn apply_filters(
    partitions: Vec<Vec<RecordBatch>>,
    filters: &[CascadeFilter],
    random_state: &RandomState,
    target_batch_rows: usize,
) -> Result<Vec<Vec<RecordBatch>>> {
    partitions
        .into_iter()
        .map(|partition| {
            let mut output = Vec::with_capacity(partition.len());
            for batch in partition {
                let mut retained = vec![true; batch.num_rows()];
                for cascade in filters {
                    let mask =
                        evaluate_membership(&batch, &cascade.keys, &cascade.filter, random_state)?;
                    for (keep, member) in retained.iter_mut().zip(mask.values()) {
                        *keep &= member;
                    }
                }
                let survivor_count = retained.iter().filter(|keep| **keep).count();
                if survivor_count == batch.num_rows() {
                    output.push(batch);
                } else if survivor_count > 0 {
                    let mask = BooleanArray::from(retained);
                    output.push(filter_record_batch(&batch, &mask)?);
                }
            }
            compact_materialized_partition(output, target_batch_rows.max(1))
        })
        .collect()
}

/// Use exact bitmap membership when possible, otherwise hash the complete
/// composite key into the probabilistic structure built by the source.
pub(super) fn evaluate_membership(
    batch: &RecordBatch,
    keys: &[Arc<dyn PhysicalExpr>],
    filter: &TransferBloomFilter,
    random_state: &RandomState,
) -> Result<BooleanArray> {
    if filter.is_dense_integer() && keys.len() == 1 {
        let array = keys[0].evaluate(batch)?.into_array(batch.num_rows())?;
        if let Some(mask) = filter.integer_mask(&array) {
            return Ok(mask);
        }
    }

    Ok(BooleanArray::from(
        evaluate_hashes(batch, keys, random_state)?
            .into_iter()
            .map(|hash| filter.might_contain(hash))
            .collect::<Vec<_>>(),
    ))
}

pub(super) fn visit_integer_values(array: &ArrayRef, mut visitor: impl FnMut(i128)) -> bool {
    macro_rules! visit {
        ($array_type:ty) => {{
            let Some(values) = array.as_any().downcast_ref::<$array_type>() else {
                return false;
            };
            for value in values.iter().flatten() {
                visitor(value as i128);
            }
            true
        }};
    }
    match array.data_type() {
        DataType::Int8 => visit!(Int8Array),
        DataType::Int16 => visit!(Int16Array),
        DataType::Int32 => visit!(Int32Array),
        DataType::Int64 => visit!(Int64Array),
        DataType::UInt8 => visit!(UInt8Array),
        DataType::UInt16 => visit!(UInt16Array),
        DataType::UInt32 => visit!(UInt32Array),
        DataType::UInt64 => visit!(UInt64Array),
        _ => false,
    }
}

pub(super) fn evaluate_hashes(
    batch: &RecordBatch,
    key_exprs: &[Arc<dyn PhysicalExpr>],
    random_state: &RandomState,
) -> Result<Vec<u64>> {
    let arrays: Vec<ArrayRef> = key_exprs
        .iter()
        .map(|expr| expr.evaluate(batch)?.into_array(batch.num_rows()))
        .collect::<Result<_>>()?;
    let mut hashes = vec![0; batch.num_rows()];
    create_hashes(arrays.iter(), random_state, &mut hashes)?;
    Ok(hashes)
}

pub(super) fn key_signature(keys: &[Arc<dyn PhysicalExpr>]) -> Vec<usize> {
    keys.iter()
        .filter_map(|key| key.downcast_ref::<Column>().map(Column::index))
        .collect()
}

pub(super) fn count_rows(partitions: &[Vec<RecordBatch>]) -> usize {
    partitions.iter().flatten().map(RecordBatch::num_rows).sum()
}
