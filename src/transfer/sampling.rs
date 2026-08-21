//! Representative source sampling for transfer estimates.
//!
//! Samples influence excitation and handoff cost estimates only. Bloom still
//! executes every committed source exactly, and formal query results never
//! depend on a sampled rowset.

use super::*;

#[derive(Debug)]
pub(super) struct SampledTable {
    pub(super) partitions: Vec<Vec<RecordBatch>>,
    pub(super) input_rows: usize,
    pub(super) output_rows: usize,
}

/// Sample the sole source of a table operator, then execute the same local
/// operator subtree over those rows. This preserves the semantics whose
/// selectivity is being estimated without materializing the full table.
pub(super) async fn sample_table(
    table: &BloomTable,
    target_rows: usize,
    mode: SamplingMode,
    instant_parquet_row_groups: usize,
    samples: &PreparedSampleCache,
    row_group_layouts: &RowGroupLayoutCache,
    context: Arc<TaskContext>,
) -> Result<Option<SampledTable>> {
    let mut sources = vec![];
    let mut path = vec![];
    collect_data_sources(&table.plan, &mut path, &mut sources);
    let [(source_path, source_plan)] = sources.as_slice() else {
        return Ok(None);
    };

    let (sampled_source, input_rows) = if let Some(source_exec) =
        source_plan.downcast_ref::<DataSourceExec>()
        && let Some(source) = source_exec
            .data_source()
            .downcast_ref::<MemorySourceConfig>()
    {
        let raw_partitions = stratified_sample(source.partitions(), target_rows);
        let input_rows = count_rows(&raw_partitions);
        let sampled_source = MemorySourceConfig::try_new_exec(
            &raw_partitions,
            source.original_schema(),
            source.projection().clone(),
        )? as Arc<dyn ExecutionPlan>;
        (sampled_source, input_rows)
    } else if let Some(parquet) = match mode {
        SamplingMode::Prepared => {
            prepared_parquet_sample(
                source_plan,
                target_rows,
                samples,
                row_group_layouts,
                Arc::clone(&context),
            )
            .await?
        }
        SamplingMode::Instant => instant_parquet_sample(
            source_plan,
            target_rows,
            instant_parquet_row_groups,
            row_group_layouts,
        )?,
    } {
        parquet
    } else {
        let partitions = source_plan.output_partitioning().partition_count().max(1);
        let per_partition = target_rows.div_ceil(partitions).max(1);
        let Some(limited_source) = source_plan.with_fetch(Some(per_partition)) else {
            return Ok(None);
        };
        let sampled =
            collect_partitioned(reset_plan_states(limited_source)?, Arc::clone(&context)).await?;
        let input_rows = count_rows(&sampled);
        let sampled_source = MemorySourceConfig::try_new_exec(&sampled, source_plan.schema(), None)?
            as Arc<dyn ExecutionPlan>;
        (sampled_source, input_rows)
    };
    let sampled_plan = replace_at_path(Arc::clone(&table.plan), source_path, sampled_source)?;
    let partitions = collect_partitioned(reset_plan_states(sampled_plan)?, context).await?;
    let output_rows = count_rows(&partitions);
    Ok(Some(SampledTable {
        partitions,
        input_rows,
        output_rows,
    }))
}

/// Build a query-local Parquet sample through the scan's existing projection
/// and predicate. Unlike a prepared sample, this avoids reading unused source
/// columns and is discarded as soon as the transfer plan has been built.
fn instant_parquet_sample(
    source_plan: &Arc<dyn ExecutionPlan>,
    target_rows: usize,
    target_row_groups: usize,
    row_group_layouts: &RowGroupLayoutCache,
) -> Result<Option<(Arc<dyn ExecutionPlan>, usize)>> {
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

    let (files, input_rows) =
        instant_sample_files(base, target_rows, target_row_groups, row_group_layouts)?;
    if files.is_empty() {
        return Err(DataFusionError::Internal(
            "Bloom instant sample has no source files".to_string(),
        ));
    }
    // Do not share the source's metrics/state with the formal scan. Preserve
    // the query predicate and projection on a fresh ParquetSource instead.
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
    Ok(Some((DataSourceExec::from_data_source(config), input_rows)))
}

/// Reuse raw Parquet rows across queries while applying each query's pushed
/// predicate and projection after the cache boundary. Cached data therefore
/// remains source-specific rather than query-specific.
async fn prepared_parquet_sample(
    source_plan: &Arc<dyn ExecutionPlan>,
    target_rows: usize,
    samples: &PreparedSampleCache,
    row_group_layouts: &RowGroupLayoutCache,
    context: Arc<TaskContext>,
) -> Result<Option<(Arc<dyn ExecutionPlan>, usize)>> {
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

    let key = prepared_parquet_sample_key(base, target_rows);
    let prepared = samples
        .get_or_try_init(key, || async {
            let (files, _) = scattered_sample_files(base, target_rows, row_group_layouts)?;
            if files.is_empty() {
                return Err(DataFusionError::Internal(
                    "Bloom prepared sample has no source files".to_string(),
                ));
            }
            let mut full_parquet = ParquetSource::new(parquet_source.table_schema().clone())
                .with_table_parquet_options(parquet_source.table_parquet_options().clone());
            if let Some(factory) = parquet_source.parquet_file_reader_factory() {
                full_parquet = full_parquet.with_parquet_file_reader_factory(Arc::clone(factory));
            }
            let full_source: Arc<dyn FileSource> = Arc::new(full_parquet);
            let full_config = FileScanConfigBuilder::from(base.clone())
                .with_source(full_source)
                .with_file_groups(
                    files
                        .into_iter()
                        .map(|file| FileGroup::new(vec![file]))
                        .collect(),
                )
                .with_limit(None)
                .with_preserve_order(true)
                .build();
            let full_plan = DataSourceExec::from_data_source(full_config);
            let schema = full_plan.schema();
            let partitions = collect_partitioned(full_plan, Arc::clone(&context)).await?;
            let input_rows = count_rows(&partitions);
            PreparedSourceSample::try_new(partitions, schema, input_rows, context.as_ref())
        })
        .await?;

    let memory =
        MemorySourceConfig::try_new_exec(&prepared.partitions, Arc::clone(&prepared.schema), None)?
            as Arc<dyn ExecutionPlan>;
    let sampled = if let Some(predicate) = parquet_source.filter() {
        Arc::new(FilterExec::try_new(predicate, memory)?) as Arc<dyn ExecutionPlan>
    } else {
        memory
    };
    let Some(projection) = parquet_source.projection() else {
        return Ok(Some((sampled, prepared.input_rows)));
    };
    let expressions = projection
        .as_ref()
        .iter()
        .map(|expression| (Arc::clone(&expression.expr), expression.alias.clone()));
    let projected =
        Arc::new(ProjectionExec::try_new(expressions, sampled)?) as Arc<dyn ExecutionPlan>;
    Ok(Some((projected, prepared.input_rows)))
}

/// Bind a reusable sample to source snapshot identity, schema, and sampling
/// target so detectable file replacement or schema evolution cannot reuse
/// stale rows.
fn prepared_parquet_sample_key(config: &FileScanConfig, target_rows: usize) -> String {
    let mut files = config
        .file_groups
        .iter()
        .flat_map(FileGroup::iter)
        .map(|file| {
            format!(
                "{}:{}:{:?}:{:?}:{:?}:{}:{}",
                file.object_meta.location,
                file.object_meta.size,
                file.object_meta.last_modified,
                file.range,
                file.partition_values,
                file.object_meta.e_tag.as_deref().unwrap_or_default(),
                file.object_meta.version.as_deref().unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();
    files.sort();
    let schema = config
        .file_source()
        .table_schema()
        .table_schema()
        .fields()
        .iter()
        .map(|field| {
            format!(
                "{}:{:?}:{}",
                field.name(),
                field.data_type(),
                field.is_nullable()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "bloom-prepared-sample-v1|{}|n={target_rows}|schema={schema}|files={}",
        config.object_store_url,
        files.join(";")
    )
}

#[derive(Debug, Clone, Copy)]
struct SampleRowGroup {
    file_index: usize,
    group_index: usize,
    rows: usize,
    ordinal: usize,
}

/// Select a small number of stratified row groups, then read one contiguous
/// window from each. Parquet pays setup and page-decode costs per touched row
/// group, so hundreds of tiny globally scattered ranges are a poor instant
/// sample even when their logical row count is small.
fn instant_sample_files(
    config: &FileScanConfig,
    target_rows: usize,
    target_row_groups: usize,
    row_group_layouts: &RowGroupLayoutCache,
) -> Result<(Vec<datafusion_datasource::PartitionedFile>, usize)> {
    let files = canonical_files(config)?;
    let mut layouts = Vec::with_capacity(files.len());
    let mut groups = Vec::new();
    let mut total_rows = 0_usize;
    for (file_index, file) in files.iter().enumerate() {
        let layout = row_group_layouts.row_group_rows(file)?;
        for (group_index, &rows) in layout.iter().enumerate() {
            let ordinal = groups.len();
            if rows > 0 {
                groups.push(SampleRowGroup {
                    file_index,
                    group_index,
                    rows,
                    ordinal,
                });
            }
            total_rows = total_rows.saturating_add(rows);
        }
        layouts.push(layout);
    }
    if total_rows == 0 {
        return Ok((files, 0));
    }
    if total_rows <= target_rows {
        return Ok((files, total_rows));
    }

    let selected_count = target_row_groups.min(target_rows).min(groups.len()).max(1);
    let seed = instant_sample_seed(&files, target_rows, target_row_groups);
    let selected = (0..selected_count)
        .map(|stratum| {
            let start = stratum * groups.len() / selected_count;
            let end = (stratum + 1) * groups.len() / selected_count;
            let width = end - start;
            let offset = (mix64(seed ^ stratum as u64) as usize) % width;
            groups[start + offset]
        })
        .collect::<Vec<_>>();
    let capacities = selected.iter().map(|group| group.rows).collect::<Vec<_>>();
    let sampled_rows = target_rows.min(capacities.iter().sum());
    let quotas = allocate_sample_quotas(&capacities, sampled_rows);

    let mut selections = layouts
        .iter()
        .map(|layout| vec![None; layout.len()])
        .collect::<Vec<Vec<Option<Vec<Range<usize>>>>>>();
    for (group, quota) in selected.iter().zip(quotas) {
        debug_assert!(quota > 0 && quota <= group.rows);
        let ranges = if quota == group.rows {
            std::iter::once(0..group.rows).collect()
        } else {
            let window_seed = mix64(seed ^ (group.ordinal as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let start = (window_seed as usize) % group.rows;
            let first_end = (start + quota).min(group.rows);
            let mut ranges = std::iter::once(start..first_end).collect::<Vec<_>>();
            if first_end - start < quota {
                ranges.push(0..(quota - (first_end - start)));
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
        let mut plan = ParquetAccessPlan::new_all(layout.len());
        for (group_index, (rows, selection)) in
            layout.iter().copied().zip(file_selections).enumerate()
        {
            match selection {
                None => plan.skip(group_index),
                Some(ranges) if ranges.len() == 1 && ranges[0].len() == rows => {
                    plan.scan(group_index)
                }
                Some(ranges) => plan.scan_selection(
                    group_index,
                    RowSelection::from_consecutive_ranges(ranges.into_iter(), rows),
                ),
            }
        }
        output.push(file.with_extension(plan));
    }
    Ok((output, sampled_rows))
}

/// Allocate the exact row target proportionally while giving every selected
/// row group at least one observation.
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
            .all(|(quota, cap)| quota <= cap)
    );
    quotas
}

fn instant_sample_seed(
    files: &[datafusion_datasource::PartitionedFile],
    target_rows: usize,
    target_row_groups: usize,
) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in files
        .iter()
        .flat_map(|file| file.object_meta.location.as_ref().as_bytes())
    {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    mix64(hash ^ target_rows as u64 ^ (target_row_groups as u64).rotate_left(32))
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

/// Spread a bounded number of short selections across the global table order.
/// This avoids prefix bias while capping the number of Parquet access points.
fn scattered_sample_files(
    config: &FileScanConfig,
    target_rows: usize,
    row_group_layouts: &RowGroupLayoutCache,
) -> Result<(Vec<datafusion_datasource::PartitionedFile>, usize)> {
    let files = canonical_files(config)?;
    let mut row_groups = Vec::with_capacity(files.len());
    let mut total_rows = 0usize;
    for file in &files {
        let groups = row_group_layouts.row_group_rows(file)?.as_ref().clone();
        total_rows = total_rows.saturating_add(groups.iter().sum::<usize>());
        row_groups.push(groups);
    }
    if total_rows == 0 {
        return Ok((files, 0));
    }
    if total_rows <= target_rows {
        return Ok((files, total_rows));
    }

    let wanted = target_rows.min(total_rows);
    let access_points = wanted.clamp(1, 256);
    let rows_per_access = wanted.div_ceil(access_points);
    let mut global_ranges = Vec::with_capacity(access_points);
    for point in 0..access_points {
        let center =
            ((2 * point + 1) as u128 * total_rows as u128 / (2 * access_points) as u128) as usize;
        let start = center
            .saturating_sub(rows_per_access / 2)
            .min(total_rows.saturating_sub(rows_per_access));
        global_ranges.push(start..(start + rows_per_access).min(total_rows));
    }
    let global_ranges = merge_ranges(global_ranges);

    let mut file_start = 0usize;
    let mut output = Vec::with_capacity(files.len());
    for (file, groups) in files.into_iter().zip(row_groups) {
        let file_rows = groups.iter().sum::<usize>();
        let file_end = file_start + file_rows;
        let mut plan = ParquetAccessPlan::new_all(groups.len());
        let mut group_start = file_start;
        for (group_index, group_rows) in groups.into_iter().enumerate() {
            let group_end = group_start + group_rows;
            let ranges = global_ranges
                .iter()
                .filter_map(|range| localize_range(range, group_start, group_end))
                .collect::<Vec<_>>();
            let selected = ranges
                .iter()
                .map(|range| range.end - range.start)
                .sum::<usize>();
            if selected == group_rows {
                plan.scan(group_index);
            } else if selected == 0 {
                plan.skip(group_index);
            } else {
                plan.scan_selection(
                    group_index,
                    RowSelection::from_consecutive_ranges(ranges.into_iter(), group_rows),
                );
            }
            group_start = group_end;
        }
        debug_assert_eq!(group_start, file_end);
        output.push(file.with_extension(plan));
        file_start = file_end;
    }
    let sampled_rows = global_ranges
        .iter()
        .map(|range| range.end - range.start)
        .sum();
    Ok((output, sampled_rows))
}

pub(super) fn localize_range(
    range: &Range<usize>,
    group_start: usize,
    group_end: usize,
) -> Option<Range<usize>> {
    let start = range.start.max(group_start);
    let end = range.end.min(group_end);
    (start < end).then(|| (start - group_start)..(end - group_start))
}

fn merge_ranges(mut ranges: Vec<Range<usize>>) -> Vec<Range<usize>> {
    ranges.sort_unstable_by_key(|range| range.start);
    let mut merged: Vec<Range<usize>> = vec![];
    for range in ranges {
        if let Some(last) = merged.last_mut()
            && range.start <= last.end
        {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    merged
}

fn collect_data_sources(
    plan: &Arc<dyn ExecutionPlan>,
    path: &mut Vec<usize>,
    output: &mut Vec<(Vec<usize>, Arc<dyn ExecutionPlan>)>,
) {
    if plan.downcast_ref::<DataSourceExec>().is_some() {
        output.push((path.clone(), Arc::clone(plan)));
        return;
    }
    for (index, child) in plan.children().into_iter().enumerate() {
        path.push(index);
        collect_data_sources(child, path, output);
        path.pop();
    }
}

/// Allocate sample capacity proportionally across in-memory partitions so one
/// large or early partition cannot dominate the estimate.
fn stratified_sample(partitions: &[Vec<RecordBatch>], target_rows: usize) -> Vec<Vec<RecordBatch>> {
    let partition_rows = partitions
        .iter()
        .map(|partition| partition.iter().map(RecordBatch::num_rows).sum::<usize>())
        .collect::<Vec<_>>();
    let total_rows = partition_rows.iter().sum::<usize>();
    if total_rows <= target_rows {
        return partitions.to_vec();
    }

    let mut quotas = partition_rows
        .iter()
        .map(|&rows| ((target_rows as u128 * rows as u128) / total_rows as u128) as usize)
        .collect::<Vec<_>>();
    let mut assigned = quotas.iter().sum::<usize>();
    while assigned < target_rows {
        let mut progressed = false;
        for (quota, &rows) in quotas.iter_mut().zip(&partition_rows) {
            if *quota < rows {
                *quota += 1;
                assigned += 1;
                progressed = true;
                if assigned == target_rows {
                    break;
                }
            }
        }
        if !progressed {
            break;
        }
    }

    partitions
        .iter()
        .zip(quotas)
        .map(|(partition, quota)| sample_partition(partition, quota))
        .collect()
}

/// Take small windows across a partition rather than a single prefix, retaining
/// coarse coverage without paying for per-row random sampling.
fn sample_partition(partition: &[RecordBatch], quota: usize) -> Vec<RecordBatch> {
    const WINDOW_ROWS: usize = 32;
    let total_rows = partition.iter().map(RecordBatch::num_rows).sum::<usize>();
    if quota == 0 {
        return vec![];
    }
    if quota >= total_rows {
        return partition.to_vec();
    }

    let window_count = quota.div_ceil(WINDOW_ROWS);
    let mut output = Vec::with_capacity(window_count + 1);
    let mut remaining = quota;
    for window in 0..window_count {
        let width = remaining.min(WINDOW_ROWS);
        let max_start = total_rows - width;
        let start = if window_count == 1 {
            max_start / 2
        } else {
            window * max_start / (window_count - 1)
        };
        append_partition_slice(partition, start, width, &mut output);
        remaining -= width;
    }
    output
}

fn append_partition_slice(
    partition: &[RecordBatch],
    start: usize,
    length: usize,
    output: &mut Vec<RecordBatch>,
) {
    let mut partition_offset = 0;
    let mut desired_start = start;
    let mut remaining = length;
    for batch in partition {
        let batch_end = partition_offset + batch.num_rows();
        if desired_start >= batch_end {
            partition_offset = batch_end;
            continue;
        }
        let local_start = desired_start.saturating_sub(partition_offset);
        let take = remaining.min(batch.num_rows() - local_start);
        output.push(batch.slice(local_start, take));
        remaining -= take;
        if remaining == 0 {
            break;
        }
        desired_start += take;
        partition_offset = batch_end;
    }
}

#[cfg(test)]
mod tests {
    use super::allocate_sample_quotas;

    #[test]
    fn parquet_sample_quotas_are_exact_and_capacity_bounded() {
        let capacities = [100, 250, 650];
        let quotas = allocate_sample_quotas(&capacities, 503);
        assert_eq!(quotas.iter().sum::<usize>(), 503);
        assert!(quotas.iter().all(|quota| *quota > 0));
        assert!(
            quotas
                .iter()
                .zip(capacities)
                .all(|(quota, cap)| *quota <= cap)
        );
    }
}
