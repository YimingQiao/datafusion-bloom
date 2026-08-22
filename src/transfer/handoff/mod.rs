//! Build and install transfer handoffs selected by the propagation scheduler.

use super::*;

mod full_rows;
use full_rows::collect_full_rows_handoff;
mod locations;
use locations::{collect_two_pass_row_location_handoff, row_location_transfer_plan};
mod predicate;
use predicate::{BloomScanBoundaryExec, CascadePredicate};
mod scan;
#[cfg(test)]
pub(super) use scan::predicate_is_expensive;
pub(super) use scan::{
    formal_transfer_scan_plan, materialization_widths, required_join_columns,
    strip_verified_local_filters,
};
use scan::{strip_parquet_source_predicates, transfer_filter_plan, transfer_scan_plan};

/// Commit a propagation source exactly once, then catch its handoff up with
/// any restrictions learned since that first materialization.
pub(super) async fn ensure_transfer_handoff(
    table: &BloomTable,
    runtime: &mut TableRuntime,
    required_columns: &[usize],
    services: &HandoffServices,
    random_state: &RandomState,
) -> Result<()> {
    if runtime.handoff.is_none() {
        let applied_filters = runtime.pending_filters.len();
        let (full_row_width, transfer_row_width) = materialization_widths(
            services.policy,
            runtime
                .sample
                .as_ref()
                .map(|sample| sample.partitions.as_slice()),
            table.plan.schema().as_ref(),
            required_columns,
        );
        runtime.handoff = Some(
            collect_transfer_handoff(
                HandoffRequest {
                    table,
                    filters: &runtime.pending_filters,
                    required_columns,
                    input_row_hint: runtime.initial_estimate.ceil() as usize,
                    locally_filtered_rows: runtime.initial_estimate.ceil() as usize,
                    expected_rows: runtime.estimated_rows.ceil() as usize,
                    full_row_width,
                    transfer_row_width,
                },
                services,
            )
            .await?,
        );
        runtime.applied_filter_count = applied_filters;
    }
    compact_handoff(
        runtime,
        random_state,
        services.context.session_config().batch_size(),
    )
}

/// Materialize a transfer result using the handoff policy, independently of
/// the scheduler that selected this source.
///
/// RowLocations is speculative: unsupported plans or poor observed locality
/// fall back to FullRows, which remains the default and correctness baseline.
pub(super) async fn collect_transfer_handoff(
    request: HandoffRequest<'_>,
    services: &HandoffServices,
) -> Result<TransferHandoff> {
    let HandoffRequest {
        table,
        filters,
        required_columns,
        input_row_hint,
        locally_filtered_rows,
        expected_rows,
        full_row_width,
        transfer_row_width,
    } = request;
    let source_rows = source_rows(&table.plan)?.unwrap_or(locally_filtered_rows);
    let facts = MaterializationFacts {
        source_rows,
        locally_filtered_rows,
        expected_rows,
        full_row_width,
        transfer_row_width,
        has_local_filter: contains_filter_exec(&table.plan),
    };
    let decision = choose_materialization(services.policy, facts);
    if services.log_steps {
        eprintln!(
            "  [handoff-policy] choice={:?} reason={} source_rows={} local_rows={} expected_rows={} full_width={} transfer_width={}",
            decision.strategy,
            decision.reason,
            facts.source_rows,
            facts.locally_filtered_rows,
            facts.expected_rows,
            facts.full_row_width,
            facts.transfer_row_width,
        );
    }

    if decision.strategy == MaterializationStrategy::TwoPassRowLocations
        && let Some(handoff) = collect_two_pass_row_location_handoff(
            table,
            filters,
            required_columns,
            input_row_hint,
            services.log_steps,
            services.parquet_layouts.as_ref(),
            Arc::clone(&services.context),
        )
        .await?
    {
        return Ok(handoff);
    }
    if decision.strategy == MaterializationStrategy::RowLocations {
        let prepare_started = Instant::now();
        if let Some((plan, original_columns, layout)) = row_location_transfer_plan(
            table,
            filters,
            required_columns,
            services.log_steps,
            services.parquet_layouts.as_ref(),
            services.context.as_ref(),
        )? {
            let prepare_elapsed = prepare_started.elapsed();
            let collect_started = Instant::now();
            let partitions =
                collect_partitioned(Arc::clone(&plan), Arc::clone(&services.context)).await?;
            let locality = layout.locality(&partitions)?;
            if services.log_steps {
                eprintln!(
                    "  [materialize-phase] mode=row-locations prepare_ms={:.3} collect_ms={:.3}",
                    prepare_elapsed.as_secs_f64() * 1000.0,
                    collect_started.elapsed().as_secs_f64() * 1000.0
                );
                eprintln!(
                    "  [materialize-metrics]\n{}",
                    DisplayableExecutionPlan::with_metrics(plan.as_ref()).indent(true)
                );
                eprintln!(
                    "  [row-location-locality] selected_rows={} runs={} touched_row_groups={}/{} touched_row_group_rows={}",
                    locality.selected_rows,
                    locality.contiguous_runs,
                    locality.touched_row_groups,
                    locality.total_row_groups,
                    locality.touched_row_group_rows,
                );
            }
            if locality.selected_rows == 0 || row_locations_are_concentrated(locality) {
                return Ok(TransferHandoff::row_locations(
                    partitions,
                    input_row_hint,
                    !filters.is_empty(),
                    original_columns,
                    layout,
                ));
            }
            if services.log_steps {
                eprintln!("  [handoff-policy] fallback=FullRows reason=scattered_row_locations");
            }
        }
    }
    if decision.strategy != MaterializationStrategy::FullRows && services.log_steps {
        eprintln!("  [handoff-policy] fallback=FullRows reason=row_location_plan_unsupported");
    }

    collect_full_rows_handoff(
        FullRowsRequest {
            table,
            filters,
            input_row_hint,
            expected_rows,
            full_row_width,
            parquet_membership_placement: services.parquet_membership_placement,
            log_transfer_steps: services.log_steps,
        },
        Arc::clone(&services.context),
    )
    .await
}

struct FullRowsRequest<'a> {
    table: &'a BloomTable,
    filters: &'a [CascadeFilter],
    input_row_hint: usize,
    expected_rows: usize,
    full_row_width: usize,
    parquet_membership_placement: ParquetMembershipPlacement,
    log_transfer_steps: bool,
}
