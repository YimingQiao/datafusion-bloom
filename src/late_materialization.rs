use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{ArrayRef, UInt32Array, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result, internal_err, plan_err};
use datafusion::datasource::physical_plan::parquet::ParquetAccessPlan;
use datafusion::datasource::physical_plan::{
    FileGroup, FileScanConfig, FileScanConfigBuilder, FileSource, ParquetSource,
};
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::TaskContext;
use datafusion::parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType, PlanProperties};
use datafusion::physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, SendableRecordBatchStream,
};
use datafusion_datasource::PartitionedFile;

use crate::handoff::RowLocationLocality;
use futures::StreamExt;

pub(crate) const FILE_ID_COLUMN: &str = "__bloom_file_id";
pub(crate) const ROW_OFFSET_COLUMN: &str = "__bloom_row_offset";

#[derive(Debug, Clone)]
struct LocatedFile {
    file: PartitionedFile,
    row_group_rows: Vec<usize>,
}

#[derive(Debug, Clone)]
pub(crate) struct RowLocationLayout {
    files: Arc<Vec<LocatedFile>>,
}

#[derive(Debug)]
pub(crate) struct PreparedLocationPlan {
    pub(crate) plan: Arc<dyn ExecutionPlan>,
    pub(crate) layout: RowLocationLayout,
}

/// Planner-owned cache for immutable Parquet row-group lengths. Stable row
/// locations need this metadata for every query, but reading all file footers
/// again is pure setup overhead.
#[derive(Debug, Default)]
pub(crate) struct PreparedRowGroupLayoutCache {
    entries: Mutex<HashMap<String, Arc<Vec<usize>>>>,
}

impl PreparedRowGroupLayoutCache {
    fn row_group_rows(&self, file: &PartitionedFile) -> Result<Arc<Vec<usize>>> {
        let key = format!(
            "{}|{}|{:?}|{:?}|{:?}",
            file.object_meta.location,
            file.object_meta.size,
            file.object_meta.last_modified,
            file.object_meta.e_tag,
            file.object_meta.version
        );
        if let Some(rows) = self
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
            return Ok(rows);
        }

        let path = local_path(file);
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?;
        let rows = Arc::new(
            reader
                .metadata()
                .row_groups()
                .iter()
                .map(|row_group| row_group.num_rows() as usize)
                .collect(),
        );
        let mut entries = self.entries.lock().map_err(|_| {
            DataFusionError::Internal("Bloom row-group layout cache lock was poisoned".to_string())
        })?;
        Ok(Arc::clone(entries.entry(key).or_insert(rows)))
    }
}

/// Attach stable `(file_id, row_offset)` columns below all local table
/// operators. This fast path is deliberately conservative: unsupported source
/// shapes fall back to ordinary Bloom collections.
pub(crate) fn try_prepare_location_plan(
    plan: Arc<dyn ExecutionPlan>,
    log_fallback: bool,
    layouts: &PreparedRowGroupLayoutCache,
) -> Result<Option<PreparedLocationPlan>> {
    let mut paths = vec![];
    collect_source_paths(&plan, &mut vec![], &mut paths);
    let [source_path] = paths.as_slice() else {
        log_location_fallback(log_fallback, &format!("sources={}", paths.len()));
        return Ok(None);
    };
    let source_plan = plan_at_path(&plan, source_path)?;
    let Some(source_exec) = source_plan.downcast_ref::<DataSourceExec>() else {
        log_location_fallback(log_fallback, "source is not DataSourceExec");
        return Ok(None);
    };
    let Some(base) = source_exec.data_source().downcast_ref::<FileScanConfig>() else {
        log_location_fallback(log_fallback, "source is not FileScanConfig");
        return Ok(None);
    };
    let Some(parquet_source) = base.file_source().downcast_ref::<ParquetSource>() else {
        log_location_fallback(log_fallback, "source is not ParquetSource");
        return Ok(None);
    };
    if !base.object_store_url.as_str().starts_with("file:") {
        log_location_fallback(log_fallback, "source is not a local file");
        return Ok(None);
    }

    let files = canonical_files(base)?;
    if files.is_empty() {
        log_location_fallback(log_fallback, "files are not canonicalizable");
        return Ok(None);
    }
    let located_files = files
        .iter()
        .map(|file| {
            let row_group_rows = layouts.row_group_rows(file)?.as_ref().clone();
            Ok::<_, DataFusionError>(LocatedFile {
                file: file.clone(),
                row_group_rows,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    // Recreate the Parquet source without pushed predicates. The physical
    // FilterExec nodes above it remain the semantic authority and now see every
    // physical row, allowing RowLocationExec to assign exact on-disk offsets.
    let mut clean_parquet = ParquetSource::new(parquet_source.table_schema().clone())
        .with_table_parquet_options(parquet_source.table_parquet_options().clone());
    if let Some(factory) = parquet_source.parquet_file_reader_factory() {
        clean_parquet = clean_parquet.with_parquet_file_reader_factory(Arc::clone(factory));
    }
    let mut clean_source: Arc<dyn FileSource> = Arc::new(clean_parquet);
    if let Some(projection) = parquet_source.projection() {
        let Some(projected) = clean_source.try_pushdown_projection(projection)? else {
            log_location_fallback(log_fallback, "source projection cannot be recreated");
            return Ok(None);
        };
        clean_source = projected;
    }

    let file_groups = files
        .iter()
        .cloned()
        .map(|file| FileGroup::new(vec![file]))
        .collect();
    let clean_config = FileScanConfigBuilder::from(base.clone())
        .with_source(clean_source)
        .with_file_groups(file_groups)
        .with_preserve_order(true)
        .build();
    let clean_exec = DataSourceExec::from_data_source(clean_config);
    let located = Arc::new(RowLocationExec::try_new(clean_exec)?) as Arc<dyn ExecutionPlan>;
    let rewritten = match append_locations_at_path(plan, source_path, located) {
        Ok(rewritten) => rewritten,
        Err(error) => {
            log_location_fallback(log_fallback, &format!("rewrite failed: {error}"));
            return Ok(None);
        }
    };

    Ok(Some(PreparedLocationPlan {
        plan: rewritten,
        layout: RowLocationLayout {
            files: Arc::new(located_files),
        },
    }))
}

fn log_location_fallback(enabled: bool, reason: &str) {
    if enabled {
        eprintln!("  [row-location] fallback reason={reason}");
    }
}

impl RowLocationLayout {
    pub(crate) fn locality(&self, partitions: &[Vec<RecordBatch>]) -> Result<RowLocationLocality> {
        let selected = collect_locations(partitions, self.files.len())?;
        let selected_rows = selected.iter().map(Vec::len).sum();
        let contiguous_runs = selected
            .iter()
            .map(|offsets| consecutive_run_count(offsets))
            .sum();
        let total_row_groups = self
            .files
            .iter()
            .map(|file| file.row_group_rows.len())
            .sum();
        let mut touched_row_groups = 0;
        let mut touched_row_group_rows = 0;
        for (file, offsets) in self.files.iter().zip(&selected) {
            let mut offset_index = 0;
            let mut group_start = 0;
            for &group_rows in &file.row_group_rows {
                let group_end = group_start + group_rows;
                let first = offset_index;
                while offset_index < offsets.len() && offsets[offset_index] < group_end {
                    offset_index += 1;
                }
                if offset_index > first {
                    touched_row_groups += 1;
                    touched_row_group_rows += group_rows;
                }
                group_start = group_end;
            }
        }
        Ok(RowLocationLocality {
            selected_rows,
            contiguous_runs,
            touched_row_groups,
            total_row_groups,
            touched_row_group_rows,
        })
    }

    pub(crate) fn rewrite_formal_plan(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        partitions: &[Vec<RecordBatch>],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let selected = collect_locations(partitions, self.files.len())?;
        self.rewrite_with_selected_locations(plan, &selected)
    }

    /// Restrict a transfer scan to an already-discovered set of physical
    /// rows, then append those original locations to its output. This lets a
    /// selective local predicate be evaluated using only its own columns
    /// before the wider join-key scan.
    pub(crate) fn rewrite_transfer_plan(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        partitions: &[Vec<RecordBatch>],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let selected = Arc::new(collect_locations(partitions, self.files.len())?);
        let selected_plan = self.rewrite_with_selected_locations(plan, selected.as_ref())?;
        if selected_plan.output_partitioning().partition_count() != selected.len() {
            return plan_err!(
                "Bloom selected row-location scan expected {} partitions, found {}",
                selected.len(),
                selected_plan.output_partitioning().partition_count()
            );
        }
        Ok(Arc::new(KnownRowLocationExec::try_new(
            selected_plan,
            selected,
        )?))
    }

    fn rewrite_with_selected_locations(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        selected: &[Vec<usize>],
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut files = Vec::with_capacity(self.files.len());
        for (located, offsets) in self.files.iter().zip(selected.iter().cloned()) {
            let access_plan = access_plan(&located.row_group_rows, offsets)?;
            files.push(located.file.clone().with_extension(access_plan));
        }

        let mut paths = vec![];
        collect_source_paths(&plan, &mut vec![], &mut paths);
        let [source_path] = paths.as_slice() else {
            return plan_err!(
                "Bloom row-location rewrite expected one data source, found {}",
                paths.len()
            );
        };
        let source_plan = plan_at_path(&plan, source_path)?;
        let Some(source_exec) = source_plan.downcast_ref::<DataSourceExec>() else {
            return internal_err!("Bloom row-location path did not resolve to DataSourceExec");
        };
        let Some(base) = source_exec.data_source().downcast_ref::<FileScanConfig>() else {
            return internal_err!("Bloom row-location source lost FileScanConfig");
        };
        let file_groups = files
            .into_iter()
            .map(|file| FileGroup::new(vec![file]))
            .collect();
        let selected_config = FileScanConfigBuilder::from(base.clone())
            .with_file_groups(file_groups)
            .build();
        let replacement = DataSourceExec::from_data_source(selected_config);
        replace_at_path(plan, source_path, replacement)
    }
}

fn consecutive_run_count(offsets: &[usize]) -> usize {
    offsets
        .iter()
        .enumerate()
        .filter(|(index, offset)| *index == 0 || offsets[*index - 1] + 1 != **offset)
        .count()
}

pub(crate) fn canonical_files(config: &FileScanConfig) -> Result<Vec<PartitionedFile>> {
    let mut by_path: BTreeMap<String, Vec<PartitionedFile>> = BTreeMap::new();
    for file in config.file_groups.iter().flat_map(FileGroup::iter) {
        // Predicate/page pruning attaches a ParquetAccessPlan to each file.
        // Row-location assignment deliberately recreates an unfiltered full
        // scan, so this particular extension must be discarded. Preserve the
        // conservative fallback for any unrelated user-defined extension.
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

pub(crate) fn local_path(file: &PartitionedFile) -> PathBuf {
    let raw = file.object_meta.location.as_ref();
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        PathBuf::from("/").join(path)
    }
}

fn collect_locations(
    partitions: &[Vec<RecordBatch>],
    file_count: usize,
) -> Result<Vec<Vec<usize>>> {
    let mut selected = vec![vec![]; file_count];
    for batch in partitions.iter().flatten() {
        if batch.num_columns() < 2 {
            return internal_err!("Bloom row-location batch lost its location columns");
        }
        let file_ids = batch
            .column(batch.num_columns() - 2)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| {
                DataFusionError::Internal("Bloom file id has an invalid type".to_string())
            })?;
        let offsets = batch
            .column(batch.num_columns() - 1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Internal("Bloom row offset has an invalid type".to_string())
            })?;
        for row in 0..batch.num_rows() {
            let file_id = file_ids.value(row) as usize;
            let Some(file_offsets) = selected.get_mut(file_id) else {
                return internal_err!("Bloom row location references unknown file {file_id}");
            };
            file_offsets.push(offsets.value(row) as usize);
        }
    }
    for offsets in &mut selected {
        offsets.sort_unstable();
        offsets.dedup();
    }
    Ok(selected)
}

fn access_plan(row_group_rows: &[usize], offsets: Vec<usize>) -> Result<ParquetAccessPlan> {
    let mut plan = ParquetAccessPlan::new_all(row_group_rows.len());
    let mut offset_index = 0;
    let mut group_start = 0;
    for (group_index, &group_rows) in row_group_rows.iter().enumerate() {
        let group_end = group_start + group_rows;
        let first = offset_index;
        while offset_index < offsets.len() && offsets[offset_index] < group_end {
            if offsets[offset_index] < group_start {
                return internal_err!("Bloom row locations are not monotonically grouped");
            }
            offset_index += 1;
        }
        let group_offsets = &offsets[first..offset_index];
        if group_offsets.len() == group_rows {
            plan.scan(group_index);
        } else if !group_offsets.is_empty() {
            let ranges =
                consecutive_ranges(group_offsets.iter().map(|offset| offset - group_start));
            plan.scan_selection(
                group_index,
                RowSelection::from_consecutive_ranges(ranges.into_iter(), group_rows),
            );
        } else {
            plan.skip(group_index);
        }
        group_start = group_end;
    }
    if offset_index != offsets.len() {
        return internal_err!("Bloom row location exceeds Parquet row count");
    }
    Ok(plan)
}

fn consecutive_ranges(offsets: impl Iterator<Item = usize>) -> Vec<std::ops::Range<usize>> {
    let mut ranges: Vec<std::ops::Range<usize>> = vec![];
    for offset in offsets {
        match ranges.last_mut() {
            Some(range) if range.end == offset => range.end += 1,
            _ => ranges.push(offset..offset + 1),
        }
    }
    ranges
}

fn collect_source_paths(
    plan: &Arc<dyn ExecutionPlan>,
    path: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if plan.downcast_ref::<DataSourceExec>().is_some() {
        output.push(path.clone());
        return;
    }
    for (index, child) in plan.children().into_iter().enumerate() {
        path.push(index);
        collect_source_paths(child, path, output);
        path.pop();
    }
}

fn plan_at_path(plan: &Arc<dyn ExecutionPlan>, path: &[usize]) -> Result<Arc<dyn ExecutionPlan>> {
    let mut current = Arc::clone(plan);
    for &index in path {
        let children = current.children();
        let Some(child) = children.get(index) else {
            return internal_err!("invalid Bloom source path {path:?}");
        };
        current = Arc::clone(child);
    }
    Ok(current)
}

fn append_locations_at_path(
    plan: Arc<dyn ExecutionPlan>,
    path: &[usize],
    replacement: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if path.is_empty() {
        return Ok(replacement);
    }
    if plan.children().len() != 1 || path[0] != 0 {
        return plan_err!("Bloom row-location fast path requires a unary table subtree");
    }
    let child = append_locations_at_path(Arc::clone(plan.children()[0]), &path[1..], replacement)?;

    if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        let mut projection = filter
            .projection()
            .as_ref()
            .map(|projection| projection.to_vec());
        if let Some(projection) = &mut projection {
            projection.push(child.schema().fields().len() - 2);
            projection.push(child.schema().fields().len() - 1);
        }
        return FilterExecBuilder::from(filter)
            .with_input(child)
            // `apply_projection` composes with the builder's existing
            // projection. The two synthetic location columns are not present
            // in that old projection, so clear it before installing the
            // extended direct projection.
            .apply_projection(None)?
            .apply_projection(projection)?
            .build()
            .map(|exec| Arc::new(exec) as Arc<dyn ExecutionPlan>);
    }
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        let mut expressions = projection.expr().to_vec();
        let width = child.schema().fields().len();
        expressions.push(
            (
                Arc::new(Column::new(FILE_ID_COLUMN, width - 2)) as _,
                FILE_ID_COLUMN.to_string(),
            )
                .into(),
        );
        expressions.push(
            (
                Arc::new(Column::new(ROW_OFFSET_COLUMN, width - 1)) as _,
                ROW_OFFSET_COLUMN.to_string(),
            )
                .into(),
        );
        return Ok(Arc::new(ProjectionExec::try_new(expressions, child)?));
    }

    // Operators such as limits, batch coalescing, and repartitioning preserve
    // their input columns. Reject any other schema-changing node.
    if plan.schema() != plan.children()[0].schema() {
        return plan_err!(
            "Bloom row-location fast path cannot preserve locations through {}",
            plan.name()
        );
    }
    plan.with_new_children(vec![child])
}

fn replace_at_path(
    plan: Arc<dyn ExecutionPlan>,
    path: &[usize],
    replacement: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if path.is_empty() {
        return Ok(replacement);
    }
    let index = path[0];
    let mut children = plan.children().into_iter().cloned().collect::<Vec<_>>();
    if index >= children.len() {
        return internal_err!("invalid Bloom replacement path {path:?}");
    }
    children[index] = replace_at_path(Arc::clone(&children[index]), &path[1..], replacement)?;
    plan.with_new_children(children)
}

struct RowLocationExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    cache: Arc<PlanProperties>,
}

struct KnownRowLocationExec {
    input: Arc<dyn ExecutionPlan>,
    locations: Arc<Vec<Vec<usize>>>,
    schema: SchemaRef,
    cache: Arc<PlanProperties>,
}

impl KnownRowLocationExec {
    fn try_new(input: Arc<dyn ExecutionPlan>, locations: Arc<Vec<Vec<usize>>>) -> Result<Self> {
        let mut fields = input.schema().fields().to_vec();
        fields.push(Arc::new(Field::new(
            FILE_ID_COLUMN,
            DataType::UInt32,
            false,
        )));
        fields.push(Arc::new(Field::new(
            ROW_OFFSET_COLUMN,
            DataType::UInt64,
            false,
        )));
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            input.schema().metadata().clone(),
        ));
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            locations,
            schema,
            cache,
        })
    }
}

impl RowLocationExec {
    fn try_new(input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        let mut fields = input.schema().fields().to_vec();
        fields.push(Arc::new(Field::new(
            FILE_ID_COLUMN,
            DataType::UInt32,
            false,
        )));
        fields.push(Arc::new(Field::new(
            ROW_OFFSET_COLUMN,
            DataType::UInt64,
            false,
        )));
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            input.schema().metadata().clone(),
        ));
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            schema,
            cache,
        })
    }
}

impl fmt::Debug for RowLocationExec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("RowLocationExec").finish()
    }
}

impl DisplayAs for RowLocationExec {
    fn fmt_as(&self, _: DisplayFormatType, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "BloomRowLocation")
    }
}

impl ExecutionPlan for RowLocationExec {
    fn name(&self) -> &'static str {
        "BloomRowLocationExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("BloomRowLocationExec expected one child");
        }
        Ok(Arc::new(Self::try_new(children.swap_remove(0))?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let schema = Arc::clone(&self.schema);
        let mut row_offset = 0_u64;
        let stream = input.map(move |batch| {
            let batch = batch?;
            let rows = batch.num_rows();
            let mut columns = batch.columns().to_vec();
            columns.push(Arc::new(UInt32Array::from_value(partition as u32, rows)) as ArrayRef);
            columns.push(Arc::new(UInt64Array::from_iter_values(
                row_offset..row_offset + rows as u64,
            )) as ArrayRef);
            row_offset += rows as u64;
            RecordBatch::try_new(Arc::clone(&schema), columns).map_err(Into::into)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }
}

impl fmt::Debug for KnownRowLocationExec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("KnownRowLocationExec").finish()
    }
}

impl DisplayAs for KnownRowLocationExec {
    fn fmt_as(&self, _: DisplayFormatType, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "BloomKnownRowLocation")
    }
}

impl ExecutionPlan for KnownRowLocationExec {
    fn name(&self) -> &'static str {
        "BloomKnownRowLocationExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("BloomKnownRowLocationExec expected one child");
        }
        Ok(Arc::new(Self::try_new(
            children.swap_remove(0),
            Arc::clone(&self.locations),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let offsets = self.locations.get(partition).cloned().ok_or_else(|| {
            DataFusionError::Internal(format!(
                "Bloom selected row-location scan references unknown partition {partition}"
            ))
        })?;
        let input = self.input.execute(partition, context)?;
        let schema = Arc::clone(&self.schema);
        let mut cursor = 0_usize;
        let stream = input.map(move |batch| {
            let batch = batch?;
            let rows = batch.num_rows();
            let end = cursor.saturating_add(rows);
            if end > offsets.len() {
                return internal_err!(
                    "Bloom selected scan emitted {} rows beyond its {} known locations",
                    end,
                    offsets.len()
                );
            }
            let mut columns = batch.columns().to_vec();
            columns.push(Arc::new(UInt32Array::from_value(partition as u32, rows)) as ArrayRef);
            columns.push(Arc::new(UInt64Array::from_iter_values(
                offsets[cursor..end].iter().map(|offset| *offset as u64),
            )) as ArrayRef);
            cursor = end;
            RecordBatch::try_new(Arc::clone(&schema), columns).map_err(Into::into)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }
}
