//! Query-local Parquet sample acquisition.
//!
//! This module chooses physical source positions and returns bounded sample
//! evidence. It does not choose propagation edges or handoff representations.

use super::super::parquet_layout::{ParquetFileLayout, ParquetLayoutCache, canonical_files};
use super::super::*;
use super::SampledTable;
use datafusion::datasource::physical_plan::parquet::{
    ParquetFileMetrics, RowGroupAccessPlanFilter,
};
use datafusion::physical_optimizer::pruning::PruningPredicate;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;

/// Acquire one dense sample from row groups that survive metadata pruning.
pub(in crate::transfer) async fn instant_parquet_sample(
    table: &BloomTable,
    source_path: &[usize],
    source_plan: &Arc<dyn ExecutionPlan>,
    target_rows: usize,
    target_row_groups: usize,
    parquet_layouts: &ParquetLayoutCache,
    context: Arc<TaskContext>,
) -> Result<Option<SampledTable>> {
    let Some(source_exec) = source_plan.downcast_ref::<DataSourceExec>() else {
        return Ok(None);
    };
    let Some(base) = source_exec.data_source().downcast_ref::<FileScanConfig>() else {
        return Ok(None);
    };
    let Some(parquet_source) = base.file_source().downcast_ref::<ParquetSource>() else {
        return Ok(None);
    };
    if !base.object_store_url.as_str().starts_with("file:") {
        return Ok(None);
    }
    let Some(sample_plan) = stratified_row_group_sample_files(
        base,
        parquet_source,
        target_rows,
        target_row_groups,
        parquet_layouts,
    )?
    else {
        return Ok(None);
    };
    if sample_plan.candidate_rows == 0 {
        return Ok(Some(SampledTable::from_candidate_sample(
            vec![],
            0,
            0,
            0,
            0,
            true,
        )));
    }
    let Some(sampled_source) = fresh_parquet_source(source_plan, sample_plan.files)? else {
        return Ok(None);
    };
    let sampled_plan = replace_at_path(Arc::clone(&table.plan), source_path, sampled_source)?;
    let partitions = collect_partitioned(reset_plan_states(sampled_plan)?, context).await?;
    let output_rows = count_rows(&partitions);
    Ok(Some(SampledTable::from_candidate_sample(
        partitions,
        sample_plan.sampled_rows,
        output_rows,
        sample_plan.candidate_rows,
        sample_plan.sampled_row_groups,
        sample_plan.sampled_rows >= sample_plan.candidate_rows,
    )))
}

#[derive(Debug, Clone, Copy)]
struct SampleRowGroup {
    file_index: usize,
    group_index: usize,
    rows: usize,
    ordinal: usize,
}

struct SampleFilePlan {
    files: Vec<datafusion_datasource::PartitionedFile>,
    sampled_rows: usize,
    candidate_rows: usize,
    sampled_row_groups: usize,
}

/// Select one row group from each candidate stratum and one contiguous
/// circular window from each selected group.
fn stratified_row_group_sample_files(
    config: &FileScanConfig,
    parquet_source: &ParquetSource,
    target_rows: usize,
    target_row_groups: usize,
    parquet_layouts: &ParquetLayoutCache,
) -> Result<Option<SampleFilePlan>> {
    let source_has_files = config
        .file_groups
        .iter()
        .any(|group| group.iter().next().is_some());
    let files = canonical_files(config)?;
    if files.is_empty() && source_has_files {
        return Ok(None);
    }
    let mut layouts = Vec::with_capacity(files.len());
    let mut groups = Vec::new();
    let mut candidate_rows = 0_usize;
    for (file_index, file) in files.iter().enumerate() {
        let layout = parquet_layouts.file_layout(file)?;
        let rows = layout.row_group_rows();
        let candidates = candidate_row_groups(parquet_source, file, &layout)?;
        for (group_index, (&rows, candidate)) in rows.iter().zip(candidates).enumerate() {
            if candidate && rows > 0 {
                groups.push(SampleRowGroup {
                    file_index,
                    group_index,
                    rows,
                    ordinal: groups.len(),
                });
                candidate_rows = candidate_rows.saturating_add(rows);
            }
        }
        layouts.push(Arc::clone(rows));
    }
    if candidate_rows == 0 {
        return Ok(Some(SampleFilePlan {
            files,
            sampled_rows: 0,
            candidate_rows: 0,
            sampled_row_groups: 0,
        }));
    }

    let population_row_groups = groups.len();
    let selected_count = target_row_groups
        .min(target_rows)
        .min(population_row_groups)
        .max(1);
    let seed = instant_sample_seed(&files, target_rows, target_row_groups);
    let selected = (0..selected_count)
        .map(|stratum| {
            let start = stratum * groups.len() / selected_count;
            let end = (stratum + 1) * groups.len() / selected_count;
            let offset = (mix64(seed ^ stratum as u64) as usize) % (end - start);
            groups[start + offset]
        })
        .collect::<Vec<_>>();
    let capacities = selected.iter().map(|group| group.rows).collect::<Vec<_>>();
    let sampled_rows = target_rows.min(candidate_rows).min(capacities.iter().sum());
    let quotas = allocate_sample_quotas(&capacities, sampled_rows);

    let mut selections = layouts
        .iter()
        .map(|layout| vec![None; layout.len()])
        .collect::<Vec<Vec<Option<Vec<Range<usize>>>>>>();
    for (group, quota) in selected.iter().zip(quotas) {
        let ranges = if quota == group.rows {
            std::iter::once(0..group.rows).collect()
        } else {
            let window_seed =
                mix64(seed ^ (group.ordinal as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let start = (window_seed as usize) % group.rows;
            let first_end = (start + quota).min(group.rows);
            let mut ranges = std::iter::once(start..first_end).collect::<Vec<_>>();
            let first_rows = first_end - start;
            if first_rows < quota {
                ranges.push(0..quota - first_rows);
                ranges.sort_unstable_by_key(|range| range.start);
            }
            ranges
        };
        selections[group.file_index][group.group_index] = Some(ranges);
    }

    let mut output = Vec::new();
    for ((file, layout), file_selections) in files.into_iter().zip(layouts).zip(selections) {
        if file_selections.iter().all(Option::is_none) {
            continue;
        }
        let plan = build_access_plan(&layout, file_selections);
        output.push(file.with_extension(plan));
    }
    Ok(Some(SampleFilePlan {
        files: output,
        sampled_rows,
        candidate_rows,
        sampled_row_groups: selected_count,
    }))
}

/// Apply the same conservative row-group statistics pruning used by the
/// formal DataFusion Parquet reader. Groups proved impossible contribute an
/// exact zero; sampling is restricted to the remaining population.
fn candidate_row_groups(
    parquet_source: &ParquetSource,
    file: &datafusion_datasource::PartitionedFile,
    layout: &ParquetFileLayout,
) -> Result<Vec<bool>> {
    let row_group_count = layout.row_group_rows().len();
    let Some(predicate) = parquet_source.filter() else {
        return Ok(vec![true; row_group_count]);
    };
    if !parquet_source.table_parquet_options().global.pruning {
        return Ok(vec![true; row_group_count]);
    }
    let Ok(predicate) = PruningPredicate::try_new(predicate, Arc::clone(layout.arrow_schema()))
    else {
        return Ok(vec![true; row_group_count]);
    };
    if predicate.always_true() {
        return Ok(vec![true; row_group_count]);
    }

    let metrics_set = ExecutionPlanMetricsSet::new();
    let metrics = ParquetFileMetrics::new(0, file.object_meta.location.as_ref(), &metrics_set);
    let mut plan = RowGroupAccessPlanFilter::new(ParquetAccessPlan::new_all(row_group_count));
    plan.prune_by_statistics(
        layout.arrow_schema().as_ref(),
        layout.metadata().file_metadata().schema_descr(),
        layout.metadata().row_groups(),
        &predicate,
        &metrics,
    );
    let mut candidates = vec![false; row_group_count];
    for index in plan.row_group_indexes() {
        candidates[index] = true;
    }
    Ok(candidates)
}

fn build_access_plan(
    row_group_rows: &[usize],
    selections: Vec<Option<Vec<Range<usize>>>>,
) -> ParquetAccessPlan {
    let mut plan = ParquetAccessPlan::new_all(row_group_rows.len());
    for (group_index, (rows, selection)) in
        row_group_rows.iter().copied().zip(selections).enumerate()
    {
        match selection {
            None => plan.skip(group_index),
            Some(ranges) if ranges.len() == 1 && ranges[0].len() == rows => plan.scan(group_index),
            Some(ranges) => plan.scan_selection(
                group_index,
                RowSelection::from_consecutive_ranges(ranges.into_iter(), rows),
            ),
        }
    }
    plan
}

fn allocate_sample_quotas(capacities: &[usize], target_rows: usize) -> Vec<usize> {
    debug_assert!(!capacities.is_empty());
    debug_assert!(target_rows >= capacities.len());
    debug_assert!(target_rows <= capacities.iter().sum());

    let mut quotas = vec![1; capacities.len()];
    let remaining = target_rows - capacities.len();
    if remaining == 0 {
        return quotas;
    }
    let extra_capacity = capacities.iter().map(|rows| rows - 1).sum::<usize>();
    let mut remainders = Vec::with_capacity(capacities.len());
    let mut assigned = 0_usize;
    for (index, &capacity) in capacities.iter().enumerate() {
        let numerator = remaining as u128 * (capacity - 1) as u128;
        let extra = (numerator / extra_capacity as u128) as usize;
        quotas[index] += extra;
        assigned += extra;
        remainders.push((numerator % extra_capacity as u128, index));
    }
    remainders.sort_unstable_by(|left, right| right.cmp(left));
    for (_, index) in remainders.into_iter().take(remaining - assigned) {
        quotas[index] += 1;
    }
    debug_assert_eq!(quotas.iter().sum::<usize>(), target_rows);
    debug_assert!(
        quotas
            .iter()
            .zip(capacities)
            .all(|(quota, capacity)| quota <= capacity)
    );
    quotas
}

fn instant_sample_seed(
    files: &[datafusion_datasource::PartitionedFile],
    target_rows: usize,
    target_row_groups: usize,
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in files
        .iter()
        .flat_map(|file| file.object_meta.location.as_ref().as_bytes())
    {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    mix64(hash ^ target_rows as u64 ^ (target_row_groups as u64).rotate_left(32))
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn fresh_parquet_source(
    source_plan: &Arc<dyn ExecutionPlan>,
    files: Vec<datafusion_datasource::PartitionedFile>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(source_exec) = source_plan.downcast_ref::<DataSourceExec>() else {
        return Ok(None);
    };
    let Some(base) = source_exec.data_source().downcast_ref::<FileScanConfig>() else {
        return Ok(None);
    };
    let Some(parquet_source) = base.file_source().downcast_ref::<ParquetSource>() else {
        return Ok(None);
    };
    if !base.object_store_url.as_str().starts_with("file:") || files.is_empty() {
        return Ok(None);
    }

    let mut fresh_parquet = ParquetSource::new(parquet_source.table_schema().clone())
        .with_table_parquet_options(parquet_source.table_parquet_options().clone());
    if let Some(factory) = parquet_source.parquet_file_reader_factory() {
        fresh_parquet = fresh_parquet.with_parquet_file_reader_factory(Arc::clone(factory));
    }
    if let Some(predicate) = parquet_source.filter() {
        fresh_parquet = fresh_parquet.with_predicate(predicate);
    }
    let mut fresh_source: Arc<dyn FileSource> = Arc::new(fresh_parquet);
    if let Some(projection) = parquet_source.projection() {
        let Some(projected) = fresh_source.try_pushdown_projection(projection)? else {
            return Ok(None);
        };
        fresh_source = projected;
    }
    let config = FileScanConfigBuilder::from(base.clone())
        .with_source(fresh_source)
        .with_file_groups(
            files
                .into_iter()
                .map(|file| FileGroup::new(vec![file]))
                .collect(),
        )
        .with_limit(None)
        .with_preserve_order(true)
        .build();
    Ok(Some(DataSourceExec::from_data_source(config)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{Field, Schema};
    use datafusion::parquet::arrow::ArrowWriter;
    use datafusion::parquet::file::properties::WriterProperties;
    use std::fs::File;

    #[test]
    fn selected_row_group_is_not_left_in_skip_state() {
        let plan = build_access_plan(
            &[1_000, 1_000, 1_000],
            vec![None, Some(std::iter::once(100..200).collect()), None],
        );
        assert!(!plan.should_scan(0));
        assert!(plan.should_scan(1));
        assert!(!plan.should_scan(2));
    }

    #[test]
    fn sample_quotas_are_exact_and_capacity_bounded() {
        let capacities = [100, 250, 650];
        let quotas = allocate_sample_quotas(&capacities, 503);
        assert_eq!(quotas.iter().sum::<usize>(), 503);
        assert!(quotas.iter().all(|quota| *quota > 0));
        assert!(
            quotas
                .iter()
                .zip(capacities)
                .all(|(quota, capacity)| *quota <= capacity)
        );
    }

    #[test]
    fn parquet_min_max_pruning_defines_the_sample_population() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("pruning.parquet");
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(
                (0..4).flat_map(|value| std::iter::repeat_n(value, 100)),
            ))],
        )?;
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(100))
            .build();
        let mut writer =
            ArrowWriter::try_new(File::create(&path)?, Arc::clone(&schema), Some(properties))?;
        writer.write(&batch)?;
        writer.close()?;

        let predicate = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("value", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int64(Some(2)))),
        )) as Arc<dyn PhysicalExpr>;
        let source = ParquetSource::new(Arc::clone(&schema)).with_predicate(predicate);
        let file =
            datafusion_datasource::PartitionedFile::from_path(path.to_string_lossy().into_owned())?;
        let layouts = ParquetLayoutCache::default();
        let layout = layouts.file_layout(&file)?;

        assert_eq!(
            candidate_row_groups(&source, &file, &layout)?,
            vec![false, false, true, false]
        );
        Ok(())
    }
}
