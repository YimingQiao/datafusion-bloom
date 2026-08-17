use std::mem::size_of;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray, StringArray,
    StringViewArray, UInt32Array,
};
use datafusion::arrow::compute::{concat_batches, take};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result};

use crate::handoff::{estimated_projection_width, estimated_schema_width, estimated_type_width};

const MIN_COMPACTION_SAVINGS_BYTES: usize = 64 * 1024;

/// Estimate handoff widths from live sample values rather than retained Arrow
/// capacity. The result informs storage policy only, never transfer scheduling.
pub(super) fn observed_handoff_widths(
    sample: Option<&[Vec<RecordBatch>]>,
    schema: &Schema,
    required_columns: &[usize],
) -> (usize, usize) {
    let Some(sample) = sample else {
        return (
            estimated_schema_width(schema),
            estimated_projection_width(schema, required_columns),
        );
    };
    let rows = count_rows(sample);
    if rows == 0 {
        return (
            estimated_schema_width(schema),
            estimated_projection_width(schema, required_columns),
        );
    }

    let full_bytes = sample
        .iter()
        .flatten()
        .flat_map(|batch| batch.columns())
        .map(logical_array_bytes)
        .sum::<usize>();
    let transfer_bytes = sample
        .iter()
        .flatten()
        .flat_map(|batch| {
            required_columns
                .iter()
                .filter_map(|&index| batch.columns().get(index))
        })
        .map(logical_array_bytes)
        .sum::<usize>();
    (
        full_bytes.div_ceil(rows).max(1),
        transfer_bytes.div_ceil(rows).saturating_add(12).max(1),
    )
}

/// Estimate bytes represented by this logical slice, not all bytes owned by
/// its backing buffers. Arrow filtering and slicing commonly retain the
/// original allocation, so `get_array_memory_size` can make a one-row sample
/// look as large as the entire source batch and incorrectly select a
/// row-location handoff.
fn logical_array_bytes(array: &ArrayRef) -> usize {
    let null_bytes = usize::from(array.null_count() > 0) * array.len().div_ceil(8);
    let value_bytes = match array.data_type() {
        DataType::Utf8 => {
            let Some(values) = array.as_any().downcast_ref::<StringArray>() else {
                return fallback_array_bytes(array, null_bytes);
            };
            (values.len() + 1).saturating_mul(size_of::<i32>())
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    .map(|index| values.value(index).len())
                    .sum::<usize>()
        }
        DataType::LargeUtf8 => {
            let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() else {
                return fallback_array_bytes(array, null_bytes);
            };
            (values.len() + 1).saturating_mul(size_of::<i64>())
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    .map(|index| values.value(index).len())
                    .sum::<usize>()
        }
        DataType::Utf8View => {
            let Some(values) = array.as_any().downcast_ref::<StringViewArray>() else {
                return fallback_array_bytes(array, null_bytes);
            };
            values.len().saturating_mul(16)
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    .map(|index| values.value(index).len().saturating_sub(12))
                    .sum::<usize>()
        }
        DataType::Binary => {
            let Some(values) = array.as_any().downcast_ref::<BinaryArray>() else {
                return fallback_array_bytes(array, null_bytes);
            };
            (values.len() + 1).saturating_mul(size_of::<i32>())
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    .map(|index| values.value(index).len())
                    .sum::<usize>()
        }
        DataType::LargeBinary => {
            let Some(values) = array.as_any().downcast_ref::<LargeBinaryArray>() else {
                return fallback_array_bytes(array, null_bytes);
            };
            (values.len() + 1).saturating_mul(size_of::<i64>())
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    .map(|index| values.value(index).len())
                    .sum::<usize>()
        }
        DataType::BinaryView => {
            let Some(values) = array.as_any().downcast_ref::<BinaryViewArray>() else {
                return fallback_array_bytes(array, null_bytes);
            };
            values.len().saturating_mul(16)
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    .map(|index| values.value(index).len().saturating_sub(12))
                    .sum::<usize>()
        }
        data_type => array.len().saturating_mul(estimated_type_width(data_type)),
    };
    value_bytes.saturating_add(null_bytes)
}

fn fallback_array_bytes(array: &ArrayRef, null_bytes: usize) -> usize {
    array
        .len()
        .saturating_mul(estimated_type_width(array.data_type()))
        .saturating_add(null_bytes)
}

/// Convert reader-produced batches into collection-owned Arrow buffers.
///
/// This is the explicit Arrow counterpart of appending selected DuckDB chunks
/// into a collection: rows are coalesced, sparse backing buffers are released,
/// and byte-view arrays receive a fresh ownership boundary.
pub(super) fn compact_materialized_partitions(
    partitions: Vec<Vec<RecordBatch>>,
    target_batch_rows: usize,
) -> Result<Vec<Vec<RecordBatch>>> {
    partitions
        .into_iter()
        .map(|partition| compact_materialized_partition(partition, target_batch_rows.max(1)))
        .collect()
}

/// Repack one partition into stable batch-sized ownership units. Compaction is
/// a materialization invariant, not merely a small-batch optimization.
pub(super) fn compact_materialized_partition(
    partition: Vec<RecordBatch>,
    target_batch_rows: usize,
) -> Result<Vec<RecordBatch>> {
    let mut output = Vec::with_capacity(partition.len());
    let mut pending = Vec::new();
    let mut pending_rows = 0_usize;

    for batch in partition.into_iter().filter(|batch| batch.num_rows() > 0) {
        let mut offset = 0;
        while offset < batch.num_rows() {
            let available = target_batch_rows - pending_rows;
            let length = available.min(batch.num_rows() - offset);
            pending.push(batch.slice(offset, length));
            pending_rows += length;
            offset += length;
            if pending_rows == target_batch_rows {
                output.push(compact_batch_group(std::mem::take(&mut pending))?);
                pending_rows = 0;
            }
        }
    }
    if !pending.is_empty() {
        output.push(compact_batch_group(pending)?);
    }
    Ok(output)
}

/// Reset physical ownership after selection: view arrays are garbage-collected
/// unconditionally, while other sparse arrays are copied only when retaining
/// their original allocation would be materially wasteful.
fn compact_batch_group(batches: Vec<RecordBatch>) -> Result<RecordBatch> {
    let batch = if let [batch] = batches.as_slice() {
        batch.clone()
    } else {
        let schema = batches
            .first()
            .map(RecordBatch::schema)
            .ok_or_else(|| DataFusionError::Internal("empty Bloom batch group".to_string()))?;
        concat_batches(&schema, &batches)?
    };

    if batch.num_columns() == 0 {
        return Ok(batch);
    }

    let row_count = u32::try_from(batch.num_rows()).map_err(|_| {
        DataFusionError::Internal(
            "Bloom compact batch exceeded UInt32 row-index capacity".to_string(),
        )
    })?;
    let indices = UInt32Array::from_iter_values(0..row_count);
    let mut changed = false;
    let columns = batch
        .columns()
        .iter()
        .map(|array| {
            match array.data_type() {
                DataType::Utf8View => {
                    changed = true;
                    let values = array
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .ok_or_else(|| {
                            DataFusionError::Internal(
                                "Utf8View array violated its physical type invariant".to_string(),
                            )
                        })?;
                    return Ok(Arc::new(values.gc()) as ArrayRef);
                }
                DataType::BinaryView => {
                    changed = true;
                    let values = array
                        .as_any()
                        .downcast_ref::<BinaryViewArray>()
                        .ok_or_else(|| {
                            DataFusionError::Internal(
                                "BinaryView array violated its physical type invariant".to_string(),
                            )
                        })?;
                    return Ok(Arc::new(values.gc()) as ArrayRef);
                }
                _ => {}
            }
            let physical = array.get_array_memory_size();
            let logical = logical_array_bytes(array);
            if physical <= logical.saturating_mul(2)
                || physical.saturating_sub(logical) < MIN_COMPACTION_SAVINGS_BYTES
            {
                return Ok(Arc::clone(array));
            }
            changed = true;
            Ok(take(array.as_ref(), &indices, None)?)
        })
        .collect::<Result<Vec<_>>>()?;
    if changed {
        Ok(RecordBatch::try_new(batch.schema(), columns)?)
    } else {
        Ok(batch)
    }
}

/// Compute the memory-pool charge for the owned representation, as opposed to
/// the logical live-byte estimate used by the handoff cost model.
pub(super) fn partition_physical_bytes(partitions: &[Vec<RecordBatch>]) -> usize {
    partitions
        .iter()
        .flatten()
        .flat_map(RecordBatch::columns)
        .map(|array| array.get_array_memory_size())
        .sum()
}

fn count_rows(partitions: &[Vec<RecordBatch>]) -> usize {
    partitions.iter().flatten().map(RecordBatch::num_rows).sum()
}
