//! Fixed-point propagation helpers used by the engine's main scheduling loop.

use super::*;

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
pub(super) fn build_transfer_filters(
    source: &TransferHandoff,
    specs: &[Vec<Arc<dyn PhysicalExpr>>],
    random_state: &RandomState,
    false_positive_rate: f64,
    context: &TaskContext,
) -> Result<Vec<Arc<TransferBloomFilter>>> {
    if specs.is_empty() {
        return Ok(vec![]);
    }

    let keys = specs
        .iter()
        .map(|keys| source.remap_keys(keys))
        .collect::<Result<Vec<_>>>()?;
    let mut integral = keys.iter().map(|keys| keys.len() == 1).collect::<Vec<_>>();
    let mut bounds: Vec<Option<(i128, i128)>> = vec![None; keys.len()];

    for batch in source.partitions().iter().flatten() {
        for (index, key) in keys.iter().enumerate() {
            if !integral[index] {
                continue;
            }
            let array = key[0].evaluate(batch)?.into_array(batch.num_rows())?;
            if !visit_integer_values(&array, |value| {
                bounds[index] = Some(match bounds[index] {
                    Some((minimum, maximum)) => (minimum.min(value), maximum.max(value)),
                    None => (value, value),
                });
            }) {
                integral[index] = false;
                bounds[index] = None;
            }
        }
    }

    let mut filters = Vec::with_capacity(bounds.len());
    for (index, bounds) in bounds.iter().enumerate() {
        let dense = if integral[index]
            && let Some((minimum, maximum)) = bounds
            && let Some(max_bits) = dense_integer_bits(*minimum, *maximum, source.row_count())
        {
            TransferBloomFilter::try_dense_integer(*minimum, *maximum, max_bits, context)?
        } else {
            None
        };
        filters.push(match dense {
            Some(filter) => filter,
            None => TransferBloomFilter::try_with_capacity(
                source.row_count(),
                false_positive_rate,
                context,
            )?,
        });
    }

    for batch in source.partitions().iter().flatten() {
        for (index, (key, filter)) in keys.iter().zip(&mut filters).enumerate() {
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
    Ok(filters.into_iter().map(Arc::new).collect())
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
            services.sampling_mode,
            services.instant_parquet_row_groups,
            samples,
            &services.row_group_layouts,
            Arc::clone(&services.context),
        )
        .await?
        {
            runtime.sample = Some(sampled.partitions);
        } else if table.repeatable {
            let partitions = table.plan.output_partitioning().partition_count().max(1);
            let per_partition = sample_rows.div_ceil(partitions).max(1);
            let limited = Arc::new(LocalLimitExec::new(Arc::clone(&table.plan), per_partition));
            runtime.sample =
                Some(collect_partitioned(limited, Arc::clone(&services.context)).await?);
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
    let sampled_rows = count_rows(sample);
    if sampled_rows == 0 {
        return Ok(runtime.initial_estimate);
    }
    let survivors = count_survivors(sample, &runtime.pending_filters, random_state)?;
    if survivors == 0 && sampled_rows as f64 + f64::EPSILON < runtime.initial_estimate {
        // A zero in a bounded sample is evidence for a very small destination,
        // not proof that the full destination is empty. Give it one sample-row
        // of weight so scheduling does not mistake sampling uncertainty for an
        // exact cardinality.
        return Ok((runtime.initial_estimate / sampled_rows as f64).max(1.0));
    }
    Ok(runtime.initial_estimate * survivors as f64 / sampled_rows as f64)
}
