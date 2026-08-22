//! Fixed-point propagation helpers used by the engine's main scheduling loop.

use super::*;
use crate::compat::is_resource_exhausted;

/// Local filters remove write contention, but each worker temporarily owns a
/// complete copy. Bound that tradeoff independently of propagation policy.
const MAX_PARALLEL_MEMBERSHIP_BYTES: usize = 64 * 1024 * 1024;
const MIN_ROWS_PER_MEMBERSHIP_WORKER: usize = 32 * 1024;

struct IntegerBounds {
    integral: Vec<bool>,
    bounds: Vec<Option<(i128, i128)>>,
}

pub(super) fn initial_active_tables(
    tables: &[TableRuntime],
    config: &BloomConfig,
) -> BTreeSet<TableId> {
    tables
        .iter()
        .enumerate()
        .filter_map(|(id, table)| {
            let reduced = table.estimated_rows < table.baseline_rows * config.excitation_threshold;
            reduced.then_some(id)
        })
        .collect()
}

/// Prefer the smallest currently useful source, matching Bloom's strategy of
/// propagating the cheapest, most selective rowset first.
pub(super) fn pop_smallest_active(
    active: &mut BTreeSet<TableId>,
    tables: &[TableRuntime],
) -> Option<TableId> {
    let best = active.iter().copied().min_by(|left, right| {
        tables[*left]
            .estimated_rows
            .total_cmp(&tables[*right].estimated_rows)
            .then_with(|| left.cmp(right))
    })?;
    active.remove(&best);
    Some(best)
}

/// Retain only adjacent directions that can carry lineage not already known at
/// the destination. This is what lets cyclic join graphs reach a fixed point.
pub(super) fn collect_outgoing_edges(
    source: TableId,
    graph: &BloomGraph,
    tables: &[TableRuntime],
    lineage: &LineageTracker,
) -> Result<Vec<DirectedEdge>> {
    graph.edges.iter().try_fold(Vec::new(), |mut output, edge| {
        if let Some(directed) = direct_edge(edge, source) {
            let destination = &tables[directed.destination];
            if destination.estimated_rows <= 0.0
                || !lineage.edge_carries_new_info(
                    source,
                    directed.destination,
                    &directed.source_keys,
                    &directed.destination_keys,
                )?
            {
                return Ok(output);
            }
            output.push(canonicalize_directed_edge(directed)?);
        }
        Ok(output)
    })
}

fn direct_edge(edge: &BloomEdge, source: TableId) -> Option<DirectedEdge> {
    if edge.left == source {
        Some(DirectedEdge {
            destination: edge.right,
            source_keys: edge.left_keys.clone(),
            destination_keys: edge.right_keys.clone(),
        })
    } else if edge.right == source {
        Some(DirectedEdge {
            destination: edge.left,
            source_keys: edge.right_keys.clone(),
            destination_keys: edge.left_keys.clone(),
        })
    } else {
        None
    }
}

/// Give a composite edge a stable key order. Equivalent transfers can then
/// share one filter build and hash the same tuple layout in both directions.
fn canonicalize_directed_edge(mut edge: DirectedEdge) -> Result<DirectedEdge> {
    if edge.source_keys.len() != edge.destination_keys.len() {
        return internal_err!(
            "Bloom edge has {} source keys and {} destination keys",
            edge.source_keys.len(),
            edge.destination_keys.len()
        );
    }
    let mut order = (0..edge.source_keys.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&index| {
        edge.source_keys[index]
            .downcast_ref::<Column>()
            .map(Column::index)
            .unwrap_or(usize::MAX)
    });
    edge.source_keys = order
        .iter()
        .map(|&index| Arc::clone(&edge.source_keys[index]))
        .collect();
    edge.destination_keys = order
        .iter()
        .map(|&index| Arc::clone(&edge.destination_keys[index]))
        .collect();
    Ok(edge)
}

/// Apply propagation received after a source was first materialized, then
/// restore the handoff's ownership invariant. FullRows reserves the temporary
/// old-plus-new peak before compaction so memory pressure triggers fallback.
pub(super) fn compact_handoff(
    runtime: &mut TableRuntime,
    random_state: &RandomState,
    target_batch_rows: usize,
) -> Result<()> {
    if runtime.applied_filter_count == runtime.pending_filters.len() {
        return Ok(());
    }
    let filters = runtime.pending_filters[runtime.applied_filter_count..].to_vec();
    let handoff = runtime
        .handoff
        .as_mut()
        .ok_or_else(|| DataFusionError::Internal("Bloom compaction without data".to_string()))?;
    let original_rows = handoff.row_count();
    let filters = handoff.remap_filters(&filters)?;
    let data = handoff.data_mut();
    if let Some(reservation) = &data.reservation {
        reservation.try_grow(partition_physical_bytes(&data.partitions))?;
    }
    data.partitions = apply_filters(
        std::mem::take(&mut data.partitions),
        &filters,
        random_state,
        target_batch_rows,
    )?;
    data.row_count = count_rows(&data.partitions);
    if let Some(reservation) = &data.reservation {
        reservation.try_resize(partition_physical_bytes(&data.partitions))?;
    }
    if data.row_count < original_rows {
        data.generation += 1;
    }
    runtime.applied_filter_count = runtime.pending_filters.len();
    Ok(())
}

/// Keep at most the strongest known lineage restriction for a given key
/// vector. Replacing an applied restriction reopens the handoff for compaction.
pub(super) fn install_cascade_filter(runtime: &mut TableRuntime, incoming: CascadeFilter) -> bool {
    for existing in &mut runtime.pending_filters {
        if key_signature(&existing.keys) != key_signature(&incoming.keys) {
            continue;
        }
        if incoming.lineage.is_subset_of(&existing.lineage) {
            return false;
        }
        if existing.lineage.is_subset_of(&incoming.lineage) {
            *existing = incoming;
            // Replacing an already-applied filter adds a stronger pending step.
            runtime.applied_filter_count = runtime
                .applied_filter_count
                .min(runtime.pending_filters.len().saturating_sub(1));
            return true;
        }
    }
    runtime.pending_filters.push(incoming);
    true
}

/// Resolve outgoing transfers against the lineage-aware cache and coalesce
/// equal key vectors into one physical filter build over the source handoff.
pub(super) fn prepare_activations(
    source: TableId,
    edges: &[DirectedEdge],
    lineage: &LineageTracker,
    cache: &TransferFilterCache,
) -> Result<PreparedActivations> {
    let mut activations = Vec::with_capacity(edges.len());
    let mut build_specs: Vec<JoinKeySpec> = vec![];
    for edge in edges {
        let snapshot = lineage.snapshot(source, &edge.source_keys)?;
        let filter = cache.get(&(source, snapshot.clone())).cloned();
        let build_index = if filter.is_none() {
            let signature = key_signature(&edge.source_keys);
            Some(
                build_specs
                    .iter()
                    .position(|keys| key_signature(keys) == signature)
                    .unwrap_or_else(|| {
                        build_specs.push(edge.source_keys.clone());
                        build_specs.len() - 1
                    }),
            )
        } else {
            None
        };
        activations.push(PreparedActivation {
            edge: edge.clone(),
            snapshot,
            filter,
            build_index,
        });
    }
    Ok((activations, build_specs))
}

/// Build every distinct outgoing structure in two shared passes over the
/// materialized source: one for integer min/max and one for insertion.
/// Physical source partitions are assigned to a bounded number of Tokio tasks,
/// so CPU work uses DataFusion's existing runtime rather than private threads.
pub(super) async fn build_transfer_filters(
    source: &TransferHandoff,
    specs: &[Vec<Arc<dyn PhysicalExpr>>],
    random_state: &RandomState,
    false_positive_rate: f64,
    context: &TaskContext,
) -> Result<(Vec<Arc<TransferBloomFilter>>, usize)> {
    if specs.is_empty() {
        return Ok((vec![], 0));
    }

    let keys = specs
        .iter()
        .map(|keys| source.remap_keys(keys))
        .collect::<Result<Vec<_>>>()?;
    let keys = Arc::new(keys);
    let max_workers = membership_worker_count(source, context);
    let IntegerBounds { integral, bounds } = if keys.iter().any(|keys| keys.len() == 1) {
        let bounds_groups = membership_work_groups(source.partitions(), max_workers);
        let partial_bounds = run_bounds_workers(bounds_groups, Arc::clone(&keys)).await?;
        merge_integer_bounds(partial_bounds, keys.len())
    } else {
        IntegerBounds {
            integral: vec![false; keys.len()],
            bounds: vec![None; keys.len()],
        }
    };

    let mut filters = allocate_transfer_filters(
        &integral,
        &bounds,
        source.row_count(),
        false_positive_rate,
        context,
    )?;
    let bytes_per_worker = filters
        .iter()
        .map(TransferBloomFilter::allocated_bytes)
        .sum::<usize>();
    let worker_count = max_workers.min(
        MAX_PARALLEL_MEMBERSHIP_BYTES
            .checked_div(bytes_per_worker.max(1))
            .unwrap_or(1)
            .max(1),
    );

    if worker_count == 1 {
        insert_membership_rows(
            source.partitions().iter().flatten(),
            keys.as_ref(),
            &mut filters,
            random_state,
            &integral,
        )?;
        return Ok((filters.into_iter().map(Arc::new).collect(), 1));
    }

    // The first set becomes the merged result. If query memory cannot support
    // worker-local copies, release any completed copies and use it serially.
    let mut worker_filters = vec![filters];
    for _ in 1..worker_count {
        match allocate_transfer_filters(
            &integral,
            &bounds,
            source.row_count(),
            false_positive_rate,
            context,
        ) {
            Ok(filters) => worker_filters.push(filters),
            Err(error) if is_resource_exhausted(&error) => {
                worker_filters.truncate(1);
                let mut filters = worker_filters.pop().expect("first filter set exists");
                insert_membership_rows(
                    source.partitions().iter().flatten(),
                    keys.as_ref(),
                    &mut filters,
                    random_state,
                    &integral,
                )?;
                return Ok((filters.into_iter().map(Arc::new).collect(), 1));
            }
            Err(error) => return Err(error),
        }
    }

    let work_groups = membership_work_groups(source.partitions(), worker_filters.len());
    let actual_workers = worker_filters.len();
    let mut tasks = tokio::task::JoinSet::new();
    for (partitions, mut filters) in work_groups.into_iter().zip(worker_filters) {
        let keys = Arc::clone(&keys);
        let random_state = random_state.clone();
        let integral = integral.clone();
        tasks.spawn(async move {
            insert_membership_rows(
                partitions.iter(),
                keys.as_ref(),
                &mut filters,
                &random_state,
                &integral,
            )?;
            Ok::<_, DataFusionError>(filters)
        });
    }
    let mut completed = Vec::with_capacity(actual_workers);
    while let Some(task) = tasks.join_next().await {
        completed.push(task.map_err(|error| DataFusionError::ExecutionJoin(Box::new(error)))??);
    }
    let mut filters = completed.remove(0);
    for worker_filters in completed {
        for (filter, worker_filter) in filters.iter_mut().zip(&worker_filters) {
            filter.merge_from(worker_filter)?;
        }
    }
    Ok((filters.into_iter().map(Arc::new).collect(), actual_workers))
}

fn allocate_transfer_filters(
    integral: &[bool],
    bounds: &[Option<(i128, i128)>],
    row_count: usize,
    false_positive_rate: f64,
    context: &TaskContext,
) -> Result<Vec<TransferBloomFilter>> {
    let mut filters = Vec::with_capacity(bounds.len());
    for (index, bounds) in bounds.iter().enumerate() {
        let dense = if integral[index]
            && let Some((minimum, maximum)) = bounds
            && let Some(max_bits) = dense_integer_bits(*minimum, *maximum, row_count)
        {
            TransferBloomFilter::try_dense_integer(*minimum, *maximum, max_bits, context)?
        } else {
            None
        };
        filters.push(match dense {
            Some(filter) => filter,
            None => {
                TransferBloomFilter::try_with_capacity(row_count, false_positive_rate, context)?
            }
        });
    }
    Ok(filters)
}

fn insert_membership_rows<'a>(
    batches: impl Iterator<Item = &'a RecordBatch>,
    keys: &[JoinKeySpec],
    filters: &mut [TransferBloomFilter],
    random_state: &RandomState,
    integral: &[bool],
) -> Result<()> {
    for batch in batches {
        for (index, (key, filter)) in keys.iter().zip(filters.iter_mut()).enumerate() {
            if filter.is_dense_integer() {
                let array = key[0].evaluate(batch)?.into_array(batch.num_rows())?;
                let supported = visit_integer_values(&array, |value| {
                    filter.insert_integer(value);
                });
                debug_assert!(supported && integral[index]);
            } else {
                for hash in evaluate_hashes(batch, key, random_state)? {
                    filter.insert(hash);
                }
            }
        }
    }
    Ok(())
}

fn inspect_integer_bounds(
    partitions: &[RecordBatch],
    keys: &[JoinKeySpec],
) -> Result<IntegerBounds> {
    let mut result = IntegerBounds {
        integral: keys.iter().map(|keys| keys.len() == 1).collect(),
        bounds: vec![None; keys.len()],
    };
    for batch in partitions {
        for (index, key) in keys.iter().enumerate() {
            if !result.integral[index] {
                continue;
            }
            let array = key[0].evaluate(batch)?.into_array(batch.num_rows())?;
            if !visit_integer_values(&array, |value| {
                result.bounds[index] = Some(match result.bounds[index] {
                    Some((minimum, maximum)) => (minimum.min(value), maximum.max(value)),
                    None => (value, value),
                });
            }) {
                result.integral[index] = false;
                result.bounds[index] = None;
            }
        }
    }
    Ok(result)
}

async fn run_bounds_workers(
    groups: Vec<Vec<RecordBatch>>,
    keys: Arc<Vec<JoinKeySpec>>,
) -> Result<Vec<IntegerBounds>> {
    if groups.len() == 1 {
        return Ok(vec![inspect_integer_bounds(&groups[0], keys.as_ref())?]);
    }
    let worker_count = groups.len();
    let mut tasks = tokio::task::JoinSet::new();
    for partitions in groups {
        let keys = Arc::clone(&keys);
        tasks.spawn(async move { inspect_integer_bounds(&partitions, keys.as_ref()) });
    }
    let mut completed = Vec::with_capacity(worker_count);
    while let Some(task) = tasks.join_next().await {
        completed.push(task.map_err(|error| DataFusionError::ExecutionJoin(Box::new(error)))??);
    }
    Ok(completed)
}

fn merge_integer_bounds(partials: Vec<IntegerBounds>, key_count: usize) -> IntegerBounds {
    let mut merged = IntegerBounds {
        integral: vec![true; key_count],
        bounds: vec![None; key_count],
    };
    for partial in partials {
        for index in 0..key_count {
            merged.integral[index] &= partial.integral[index];
            if !merged.integral[index] {
                merged.bounds[index] = None;
                continue;
            }
            if let Some((minimum, maximum)) = partial.bounds[index] {
                merged.bounds[index] = Some(match merged.bounds[index] {
                    Some((merged_minimum, merged_maximum)) => {
                        (merged_minimum.min(minimum), merged_maximum.max(maximum))
                    }
                    None => (minimum, maximum),
                });
            }
        }
    }
    merged
}

fn membership_worker_count(source: &TransferHandoff, context: &TaskContext) -> usize {
    if !tokio::runtime::Handle::try_current()
        .is_ok_and(|handle| handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread)
    {
        return 1;
    }
    let non_empty_partitions = source
        .partitions()
        .iter()
        .filter(|partition| !partition.is_empty())
        .count();
    let row_workers = (source.row_count() / MIN_ROWS_PER_MEMBERSHIP_WORKER).max(1);
    context
        .session_config()
        .target_partitions()
        .min(non_empty_partitions)
        .min(row_workers)
        .max(1)
}

/// Greedily assign complete execution partitions by row count. RecordBatch
/// clones only retain Arrow buffers; membership workers never mutate handoff
/// data and do not duplicate its materialized payload.
fn membership_work_groups(
    partitions: &[Vec<RecordBatch>],
    worker_count: usize,
) -> Vec<Vec<RecordBatch>> {
    let mut indexed = partitions
        .iter()
        .filter(|partition| !partition.is_empty())
        .map(|partition| {
            (
                partition.iter().map(RecordBatch::num_rows).sum::<usize>(),
                partition,
            )
        })
        .collect::<Vec<_>>();
    indexed.sort_unstable_by_key(|item| std::cmp::Reverse(item.0));

    let worker_count = worker_count.min(indexed.len()).max(1);
    let mut groups = vec![Vec::new(); worker_count];
    let mut group_rows = vec![0_usize; worker_count];
    for (rows, partition) in indexed {
        let target = group_rows
            .iter()
            .enumerate()
            .min_by_key(|(_, rows)| **rows)
            .map(|(index, _)| index)
            .expect("at least one membership worker");
        groups[target].extend(partition.iter().cloned());
        group_rows[target] = group_rows[target].saturating_add(rows);
    }
    groups
}

/// Admit exact integer membership only for a bounded or sufficiently dense
/// domain; sparse wide domains stay on the fixed-size probabilistic path.
fn dense_integer_bits(minimum: i128, maximum: i128, rows: usize) -> Option<usize> {
    let range = maximum.checked_sub(minimum)?;
    let span = range.checked_add(1)?;
    let density_limit = (rows as i128).saturating_mul(128);
    if range > 8_000_000 && range > density_limit {
        return None;
    }
    usize::try_from(span).ok()
}

/// Estimate the cardinality after all current transfers without changing the
/// formal handoff policy. Existing handoffs yield exact survivor counts;
/// otherwise a bounded sample drives scheduling only.
pub(super) async fn estimate_destination(
    table: &BloomTable,
    runtime: &mut TableRuntime,
    required_columns: &[usize],
    services: &HandoffServices,
    random_state: &RandomState,
    sample_rows: usize,
    samples: &PreparedSampleCache,
) -> Result<f64> {
    if let Some(handoff) = &runtime.handoff {
        let unapplied = &runtime.pending_filters[runtime.applied_filter_count..];
        let unapplied = handoff.remap_filters(unapplied)?;
        return Ok(count_survivors(handoff.partitions(), &unapplied, random_state)? as f64);
    }

    if runtime.sample.is_none() {
        if let Some(sampled) = sample_table(
            table,
            sample_rows,
            SamplingOptions {
                mode: services.sampling_mode,
                instant_parquet_row_groups: services.instant_parquet_row_groups,
                log_steps: services.log_steps,
            },
            samples,
            &services.parquet_layouts,
            Arc::clone(&services.context),
        )
        .await?
        {
            runtime.sample = Some(sampled);
        } else if table.repeatable {
            let partitions = table.plan.output_partitioning().partition_count().max(1);
            let per_partition = sample_rows.div_ceil(partitions).max(1);
            let limited = Arc::new(LocalLimitExec::new(Arc::clone(&table.plan), per_partition));
            runtime.sample = Some(SampledTable::from_output_partitions(
                collect_partitioned(limited, Arc::clone(&services.context)).await?,
            ));
        } else {
            // An unusual bounded source without a clonable fetch path cannot
            // be sampled safely. Execute it once and retain the exact result.
            ensure_transfer_handoff(table, runtime, required_columns, services, random_state)
                .await?;
            return Ok(runtime
                .handoff
                .as_ref()
                .map_or(0.0, |handoff| handoff.row_count() as f64));
        }
    }
    let sample = runtime.sample.as_ref().ok_or_else(|| {
        DataFusionError::Internal("Bloom destination sample was not initialized".to_string())
    })?;
    if sample.output_rows == 0 {
        return Ok(runtime.initial_estimate);
    }
    let survivors = count_survivors(&sample.partitions, &runtime.pending_filters, random_state)?;
    if survivors == 0
        && !sample.is_exact()
        && sample.output_rows as f64 + f64::EPSILON < runtime.initial_estimate
    {
        // Acquisition is complete before propagation. A transfer zero means
        // "smaller than one observed row", not a reason to perform more I/O.
        return Ok((runtime.initial_estimate / sample.output_rows as f64).max(1.0));
    }
    Ok(sample.estimate_transfer_survivors(survivors, runtime.initial_estimate))
}
