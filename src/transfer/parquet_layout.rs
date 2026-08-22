//! Shared physical layout helpers for local Parquet sources.
//!
//! Footer metadata is immutable source state. Sampling and optional
//! row-location handoffs may interpret it differently, but neither owns it or
//! stores query predicates in the cache.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::physical_plan::parquet::ParquetAccessPlan;
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig};
use datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use datafusion::parquet::file::metadata::ParquetMetaData;
use datafusion_datasource::PartitionedFile;

/// Immutable footer information shared across physical transfer policies.
#[derive(Debug)]
pub(super) struct ParquetFileLayout {
    row_group_rows: Arc<Vec<usize>>,
    metadata: Arc<ParquetMetaData>,
    arrow_schema: SchemaRef,
}

impl ParquetFileLayout {
    pub(super) fn row_group_rows(&self) -> &Arc<Vec<usize>> {
        &self.row_group_rows
    }

    pub(super) fn metadata(&self) -> &Arc<ParquetMetaData> {
        &self.metadata
    }

    pub(super) fn arrow_schema(&self) -> &SchemaRef {
        &self.arrow_schema
    }
}

/// Planner-owned cache for immutable Parquet footer metadata.
#[derive(Debug, Default)]
pub(crate) struct ParquetLayoutCache {
    entries: Mutex<HashMap<String, Arc<ParquetFileLayout>>>,
}

impl ParquetLayoutCache {
    /// Cache footer metadata under source snapshot identity. Query predicates
    /// are applied after this boundary and are never retained.
    pub(super) fn file_layout(&self, file: &PartitionedFile) -> Result<Arc<ParquetFileLayout>> {
        let key = format!(
            "{}|{}|{:?}|{:?}|{:?}",
            file.object_meta.location,
            file.object_meta.size,
            file.object_meta.last_modified,
            file.object_meta.e_tag,
            file.object_meta.version
        );
        if let Some(layout) = self
            .entries
            .lock()
            .map_err(|_| {
                DataFusionError::Internal(
                    "Bloom row-group layout cache lock was poisoned".to_string(),
                )
            })?
            .get(&key)
            .cloned()
        {
            return Ok(layout);
        }

        let path = local_path(file);
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
        let metadata = Arc::clone(reader.metadata());
        let row_group_rows = Arc::new(
            metadata
                .row_groups()
                .iter()
                .map(|row_group| row_group.num_rows() as usize)
                .collect(),
        );
        let layout = Arc::new(ParquetFileLayout {
            row_group_rows,
            metadata,
            arrow_schema: Arc::clone(reader.schema()),
        });
        let mut entries = self.entries.lock().map_err(|_| {
            DataFusionError::Internal("Bloom row-group layout cache lock was poisoned".to_string())
        })?;
        Ok(Arc::clone(entries.entry(key).or_insert(layout)))
    }

    pub(super) fn row_group_rows(&self, file: &PartitionedFile) -> Result<Arc<Vec<usize>>> {
        Ok(Arc::clone(self.file_layout(file)?.row_group_rows()))
    }
}

/// Reconstruct one canonical whole-file entry from DataFusion's possibly split
/// scan groups. Gaps and unknown extensions make physical offsets unsafe.
pub(super) fn canonical_files(config: &FileScanConfig) -> Result<Vec<PartitionedFile>> {
    let mut by_path: BTreeMap<String, Vec<PartitionedFile>> = BTreeMap::new();
    for file in config.file_groups.iter().flat_map(FileGroup::iter) {
        // DataFusion may attach a Parquet access plan after predicate pruning.
        // The caller needs the canonical file and constructs its own plan.
        let has_only_access_plan =
            file.extensions.len() == 1 && file.extensions.get::<ParquetAccessPlan>().is_some();
        if !file.extensions.is_empty() && !has_only_access_plan {
            return Ok(vec![]);
        }
        by_path
            .entry(file.object_meta.location.to_string())
            .or_default()
            .push(file.clone());
    }

    let mut output = Vec::with_capacity(by_path.len());
    for parts in by_path.into_values() {
        let mut file = parts[0].clone();
        let size = file.object_meta.size;
        let covers_full_file =
            parts.iter().any(|part| part.range.is_none()) || ranges_cover_file(&parts, size);
        if !covers_full_file {
            return Ok(vec![]);
        }
        file.range = None;
        file.extensions = Default::default();
        output.push(file);
    }
    Ok(output)
}

fn ranges_cover_file(parts: &[PartitionedFile], size: u64) -> bool {
    let mut ranges = parts
        .iter()
        .filter_map(|part| part.range.as_ref())
        .map(|range| (range.start.max(0) as u64, range.end.max(0) as u64))
        .collect::<Vec<_>>();
    ranges.sort_unstable();
    let mut covered = 0_u64;
    for (start, end) in ranges {
        if start > covered {
            return false;
        }
        covered = covered.max(end);
    }
    covered >= size
}

fn local_path(file: &PartitionedFile) -> PathBuf {
    let raw = file.object_meta.location.as_ref();
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        PathBuf::from("/").join(path)
    }
}
