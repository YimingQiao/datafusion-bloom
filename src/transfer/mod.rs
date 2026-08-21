//! Bloom's query-local transfer phase.
//!
//! The scheduler in this module reasons only about graph propagation and
//! cardinality. Its supporting modules keep the physical concerns separate:
//! sampling estimates reductions, policy chooses a handoff representation,
//! handoff builds it, materialization establishes owned Arrow buffers, and
//! row_locations implements the optional late-materialization path.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use datafusion::arrow::array::{
    Array, ArrayRef, BooleanArray, Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use datafusion::arrow::compute::filter_record_batch;
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::config::ConfigOptions;
use datafusion::common::hash_utils::{RandomState, create_hashes};
use datafusion::common::{DataFusionError, Result, ScalarValue, Statistics, internal_err};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::physical_plan::parquet::ParquetAccessPlan;
use datafusion::datasource::physical_plan::{
    FileGroup, FileScanConfig, FileScanConfigBuilder, FileSource, ParquetSource,
};
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::logical_expr::{ColumnarValue, Operator};
use datafusion::parquet::arrow::arrow_reader::RowSelection;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_expr::utils::{conjunction, split_conjunction};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::filter_pushdown::FilterPushdown as FilterPushdownRule;
use datafusion::physical_optimizer::projection_pushdown::ProjectionPushdown;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::execution_plan::{
    collect_partitioned, execute_stream_partitioned, reset_plan_states,
};
use datafusion::physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterPushdownResult, ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation,
    PushedDown,
};
use datafusion::physical_plan::joins::{
    CrossJoinExec, HashJoinExec, NestedLoopJoinExec, PartitionMode, PiecewiseMergeJoinExec,
    SortMergeJoinExec, SymmetricHashJoinExec,
};
use datafusion::physical_plan::limit::LocalLimitExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::scalar_subquery::ScalarSubqueryExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream, with_new_children_if_necessary,
};
use futures::TryStreamExt;
use futures::future::try_join_all;

use self::materialization::{
    MaterializedPartitionBuilder, batch_physical_bytes, compact_materialized_partition,
    observed_handoff_widths, partition_physical_bytes,
};
use self::policy::{
    MaterializationFacts, MaterializationStrategy, choose_materialization,
    estimated_projection_width, estimated_schema_width, row_locations_are_concentrated,
};
use self::row_locations::{RowLocationLayout, canonical_files, try_prepare_location_plan};
use self::sample_cache::PreparedSourceSample;
use crate::collection::BloomCollection;
use crate::config::{BloomConfig, HandoffPolicy, ParquetMembershipPlacement, SamplingMode};
use crate::filter::TransferBloomFilter;
use crate::graph::{BloomEdge, BloomGraph, BloomTable, TableId};
use crate::lineage::{LineageSnapshot, LineageTracker};

// Physical transfer lifecycle. These modules must not feed storage-specific
// facts back into the propagation scheduler above.
mod handoff;
use handoff::{
    collect_transfer_handoff, ensure_transfer_handoff, formal_transfer_scan_plan,
    materialization_widths, required_join_columns, strip_verified_local_filters,
};
mod inspection;
use inspection::*;
mod materialization;
mod membership;
use membership::*;
mod policy;
mod propagation;
use propagation::*;
mod row_locations;
mod sample_cache;
mod sampling;
use sampling::sample_table;

pub(crate) use row_locations::RowGroupLayoutCache;
pub(crate) use sample_cache::PreparedSampleCache;

const HASH_SEED: u64 = 0x424c_4f4f_4d44_4631;

#[derive(Debug)]
pub(crate) struct BloomTransferEngine {
    config: BloomConfig,
    samples: Arc<PreparedSampleCache>,
    row_group_layouts: Arc<RowGroupLayoutCache>,
}

#[derive(Debug)]
struct HandoffData {
    partitions: Vec<Vec<RecordBatch>>,
    input_row_count: usize,
    row_count: usize,
    generation: u64,
    reservation: Option<MemoryReservation>,
}

/// Query-local result of committing a table as a transfer source.
///
/// The scheduler only reasons about cardinality and lineage. This enum records
/// the independently chosen physical handoff: either owned query columns or
/// stable source positions plus the transfer columns needed by later rounds.
#[derive(Debug)]
enum TransferHandoff {
    FullRows(HandoffData),
    RowLocations {
        data: HandoffData,
        original_columns: Vec<usize>,
        layout: RowLocationLayout,
    },
}

impl TransferHandoff {
    fn full_rows(
        partitions: Vec<Vec<RecordBatch>>,
        input_row_hint: usize,
        filtered: bool,
        reservation: MemoryReservation,
    ) -> Self {
        let row_count = count_rows(&partitions);
        Self::FullRows(HandoffData {
            partitions,
            input_row_count: input_row_hint.max(row_count),
            row_count,
            generation: u64::from(filtered),
            reservation: Some(reservation),
        })
    }

    fn row_locations(
        partitions: Vec<Vec<RecordBatch>>,
        input_row_hint: usize,
        filtered: bool,
        original_columns: Vec<usize>,
        layout: RowLocationLayout,
    ) -> Self {
        let row_count = count_rows(&partitions);
        Self::RowLocations {
            data: HandoffData {
                partitions,
                input_row_count: input_row_hint.max(row_count),
                row_count,
                generation: u64::from(filtered),
                reservation: None,
            },
            original_columns,
            layout,
        }
    }

    fn data(&self) -> &HandoffData {
        match self {
            Self::FullRows(data) | Self::RowLocations { data, .. } => data,
        }
    }

    fn data_mut(&mut self) -> &mut HandoffData {
        match self {
            Self::FullRows(data) | Self::RowLocations { data, .. } => data,
        }
    }

    fn row_count(&self) -> usize {
        self.data().row_count
    }

    fn partitions(&self) -> &[Vec<RecordBatch>] {
        &self.data().partitions
    }

    fn remap_keys(&self, keys: &[Arc<dyn PhysicalExpr>]) -> Result<Vec<Arc<dyn PhysicalExpr>>> {
        let Self::RowLocations {
            original_columns, ..
        } = self
        else {
            return Ok(keys.to_vec());
        };
        keys.iter()
            .map(|key| {
                let column = key.downcast_ref::<Column>().ok_or_else(|| {
                    DataFusionError::Internal(
                        "Bloom row-location materialization requires column join keys".to_string(),
                    )
                })?;
                let position = original_columns
                    .iter()
                    .position(|index| *index == column.index())
                    .ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "Bloom row-location materialization omitted join column {}",
                            column.index()
                        ))
                    })?;
                Ok(Arc::new(Column::new(column.name(), position)) as Arc<dyn PhysicalExpr>)
            })
            .collect()
    }

    fn remap_filters(&self, filters: &[CascadeFilter]) -> Result<Vec<CascadeFilter>> {
        filters
            .iter()
            .map(|filter| {
                Ok(CascadeFilter {
                    keys: self.remap_keys(&filter.keys)?,
                    filter: Arc::clone(&filter.filter),
                    lineage: filter.lineage.clone(),
                })
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct CascadeFilter {
    keys: Vec<Arc<dyn PhysicalExpr>>,
    filter: Arc<TransferBloomFilter>,
    lineage: LineageSnapshot,
}

#[derive(Debug)]
struct TableRuntime {
    initial_estimate: f64,
    estimated_rows: f64,
    baseline_rows: f64,
    pending_filters: Vec<CascadeFilter>,
    applied_filter_count: usize,
    sample: Option<Vec<Vec<RecordBatch>>>,
    handoff: Option<TransferHandoff>,
}

/// Runtime services owned by the handoff/materialization layer. Keeping these
/// together makes the scheduler submit table facts without embedding storage
/// policy in its propagation decisions.
#[derive(Clone)]
struct HandoffServices {
    policy: HandoffPolicy,
    parquet_membership_placement: ParquetMembershipPlacement,
    sampling_mode: SamplingMode,
    instant_parquet_row_groups: usize,
    log_steps: bool,
    row_group_layouts: Arc<RowGroupLayoutCache>,
    context: Arc<TaskContext>,
}

struct HandoffRequest<'a> {
    table: &'a BloomTable,
    filters: &'a [CascadeFilter],
    required_columns: &'a [usize],
    input_row_hint: usize,
    locally_filtered_rows: usize,
    expected_rows: usize,
    full_row_width: usize,
    transfer_row_width: usize,
}

#[derive(Debug, Clone)]
struct DirectedEdge {
    destination: TableId,
    source_keys: Vec<Arc<dyn PhysicalExpr>>,
    destination_keys: Vec<Arc<dyn PhysicalExpr>>,
}

#[derive(Debug)]
struct PreparedActivation {
    edge: DirectedEdge,
    snapshot: LineageSnapshot,
    filter: Option<Arc<TransferBloomFilter>>,
    build_index: Option<usize>,
}

type TransferFilterCache = HashMap<(TableId, LineageSnapshot), Arc<TransferBloomFilter>>;
type JoinKeySpec = Vec<Arc<dyn PhysicalExpr>>;
type PreparedActivations = (Vec<PreparedActivation>, Vec<JoinKeySpec>);
type RowLocationTransferPlan = (Arc<dyn ExecutionPlan>, Vec<usize>, RowLocationLayout);

impl BloomTransferEngine {
    pub(crate) fn new(
        config: BloomConfig,
        samples: Arc<PreparedSampleCache>,
        row_group_layouts: Arc<RowGroupLayoutCache>,
    ) -> Self {
        Self {
            config,
            samples,
            row_group_layouts,
        }
    }

    /// Run Bloom's transfer fixed point and splice its final handoffs into an
    /// independently planned native join tree.
    ///
    /// Temporary membership structures and samples guide P0 only. They enter
    /// formal execution solely through the explicit FullRows, RowLocations, or
    /// Direct handoff selected during finalization.
    pub(crate) async fn rewrite(
        &self,
        transfer_plan: Arc<dyn ExecutionPlan>,
        formal_plan: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let transfer_started = Instant::now();
        // Scalar-subquery expressions are populated by ScalarSubqueryExec at
        // formal execution time. Executing one of its descendants as an
        // independent transfer table would observe the still-pending scalar.
        // Keep the complete scope on DataFusion's native execution path.
        if contains_scalar_subquery(&transfer_plan) {
            if self.config.log_transfer_steps {
                eprintln!("[Bloom] scope skipped: scalar_subquery_dependency");
            }
            return Ok(formal_plan);
        }
        if self.config.handoff_policy == HandoffPolicy::FullRows
            && context.session_config().target_partitions() > 1
            && context
                .session_config()
                .options()
                .optimizer
                .enable_join_dynamic_filter_pushdown
            && let Some(coverage) = native_join_filter_coverage(&formal_plan)
            && coverage.collect_left.saturating_mul(3) >= coverage.join_count.saturating_mul(2)
        {
            // DataFusion's CollectLeft joins build a dynamic filter from the
            // left input before scanning the right input. When native formal
            // execution already covers most join boundaries that way, running
            // every reduction table by table in transfer serializes largely
            // duplicate work. Keep Bloom for scopes where Partitioned joins
            // leave a substantial part of the graph uncovered. This is the
            // DataFusion counterpart of Bloom's DuckDB left-deep guard.
            if self.config.log_transfer_steps {
                eprintln!(
                    "[Bloom] scope skipped: native_join_filter_coverage collect_left={} joins={}",
                    coverage.collect_left, coverage.join_count
                );
            }
            return Ok(formal_plan);
        }
        let Some(graph) = BloomGraph::build(&transfer_plan, &self.config)? else {
            return Ok(formal_plan);
        };
        let initialize_started = Instant::now();
        let mut tables = initialize_tables(
            &graph,
            &self.config,
            Arc::clone(&self.samples),
            Arc::clone(&self.row_group_layouts),
            Arc::clone(&context),
        )
        .await?;
        if self.config.log_transfer_steps {
            eprintln!(
                "[Bloom] initialized tables={} edges={} elapsed_ms={:.3}",
                graph.tables.len(),
                graph.edges.len(),
                initialize_started.elapsed().as_secs_f64() * 1000.0
            );
            for (table, runtime) in tables.iter().enumerate() {
                eprintln!(
                    "  [table {table}] estimate={:.0} baseline={:.0} sample_rows={}",
                    runtime.estimated_rows,
                    runtime.baseline_rows,
                    runtime
                        .sample
                        .as_ref()
                        .map_or(0, |sample| count_rows(sample))
                );
            }
        }
        let mut lineage = LineageTracker::try_new(&graph)?;
        let mut filter_cache = TransferFilterCache::new();
        let mut active = initial_active_tables(&tables, &self.config);
        let random_state = RandomState::with_seed(HASH_SEED);
        let handoff_services = HandoffServices {
            policy: self.config.handoff_policy,
            parquet_membership_placement: self.config.parquet_membership_placement,
            sampling_mode: self.config.sampling_mode,
            instant_parquet_row_groups: self.config.instant_parquet_row_groups,
            log_steps: self.config.log_transfer_steps,
            row_group_layouts: Arc::clone(&self.row_group_layouts),
            context: Arc::clone(&context),
        };

        // Adaptive excitation follows Bloom's control flow: pick the smallest
        // useful source, execute it exactly, transfer its current lineage, and
        // use a bounded destination sample before deciding what to execute next.
        for round in 0..self.config.max_transfer_rounds {
            let Some(source) = pop_smallest_active(&mut active, &tables) else {
                break;
            };
            let candidates = collect_outgoing_edges(source, &graph, &tables, &lineage)?;
            if candidates.is_empty() {
                if self.config.log_transfer_steps {
                    eprintln!(
                        "[Bloom] round={} source={} no informative edges",
                        round + 1,
                        source
                    );
                }
                continue;
            }
            if self.config.log_transfer_steps {
                eprintln!(
                    "[Bloom] round={} source={} estimate={:.0} candidates={}",
                    round + 1,
                    source,
                    tables[source].estimated_rows,
                    candidates.len()
                );
            }

            // Match Bloom's excitation semantics: committing a source advances
            // its comparison baseline to the estimate that caused the
            // excitation. The exact materialized cardinality is execution data,
            // not a replacement for the estimator's state.
            let committed_estimate = tables[source].estimated_rows;
            let required_columns = required_join_columns(&graph, source)?;
            let handoff_started = Instant::now();
            ensure_transfer_handoff(
                &graph.tables[source],
                &mut tables[source],
                &required_columns,
                &handoff_services,
                &random_state,
            )
            .await?;
            if tables[source].handoff.is_none() {
                return Err(DataFusionError::Internal(format!(
                    "missing Bloom materialization for source {source}"
                )));
            }
            tables[source].baseline_rows = committed_estimate;
            if self.config.log_transfer_steps {
                let handoff = tables[source].handoff.as_ref().ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "missing Bloom materialization for logged source {source}"
                    ))
                })?;
                eprintln!(
                    "  [handoff] source={} kind={} rows={} columns={} elapsed_ms={:.3}",
                    source,
                    match handoff {
                        TransferHandoff::FullRows(_) => "FullRows",
                        TransferHandoff::RowLocations { .. } => "RowLocations",
                    },
                    handoff.row_count(),
                    handoff.partitions().iter().flatten().next().map_or(
                        graph.tables[source].plan.schema().fields().len(),
                        RecordBatch::num_columns
                    ),
                    handoff_started.elapsed().as_secs_f64() * 1000.0
                );
            }

            // This is an exact table-operator result, not a sampling estimate.
            // An empty input makes its inner-join subtree empty, so scanning
            // pending destinations cannot improve formal execution.
            if tables[source]
                .handoff
                .as_ref()
                .is_some_and(|handoff| handoff.row_count() == 0)
            {
                for table in &mut tables {
                    table.pending_filters.clear();
                    table.applied_filter_count = 0;
                }
                if self.config.log_transfer_steps {
                    eprintln!("  [empty] source={} stop transfer", source);
                }
                break;
            }

            let build_started = Instant::now();
            let (mut activations, build_specs) =
                prepare_activations(source, &candidates, &lineage, &filter_cache)?;
            let handoff = tables[source].handoff.as_ref().ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "missing Bloom materialization for filter source {source}"
                ))
            })?;
            let built = build_transfer_filters(
                handoff,
                &build_specs,
                &random_state,
                self.config.false_positive_rate,
                handoff_services.context.as_ref(),
            )?;
            if self.config.log_transfer_steps {
                eprintln!(
                    "  [build] distinct={} cache_hits={} elapsed_ms={:.3}",
                    build_specs.len(),
                    activations
                        .iter()
                        .filter(|activation| activation.filter.is_some())
                        .count(),
                    build_started.elapsed().as_secs_f64() * 1000.0
                );
            }
            let mut affected = BTreeMap::new();
            for activation in &mut activations {
                let filter = if let Some(filter) = &activation.filter {
                    Arc::clone(filter)
                } else {
                    let index = activation.build_index.ok_or_else(|| {
                        DataFusionError::Internal(
                            "Bloom activation has neither a cached nor built filter".to_string(),
                        )
                    })?;
                    let filter = Arc::clone(&built[index]);
                    filter_cache.insert((source, activation.snapshot.clone()), Arc::clone(&filter));
                    filter
                };
                let cascade = CascadeFilter {
                    keys: activation.edge.destination_keys.clone(),
                    filter,
                    lineage: activation.snapshot.clone(),
                };
                if install_cascade_filter(&mut tables[activation.edge.destination], cascade) {
                    affected
                        .entry(activation.edge.destination)
                        .or_insert_with(|| activation.edge.clone());
                }
            }

            for edge in affected.values() {
                lineage.propagate(source, edge.destination, &edge.source_keys)?;
            }

            for destination in affected.into_keys() {
                let estimate_started = Instant::now();
                let required_columns = required_join_columns(&graph, destination)?;
                let estimate = estimate_destination(
                    &graph.tables[destination],
                    &mut tables[destination],
                    &required_columns,
                    &handoff_services,
                    &random_state,
                    self.config.sample_rows,
                    &self.samples,
                )
                .await?;
                tables[destination].estimated_rows = estimate;
                if self.config.log_transfer_steps {
                    eprintln!(
                        "  [estimate] destination={} rows={:.0} filters={} elapsed_ms={:.3}",
                        destination,
                        estimate,
                        tables[destination].pending_filters.len(),
                        estimate_started.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let reduced =
                    estimate < tables[destination].baseline_rows * self.config.excitation_threshold;
                if reduced {
                    active.insert(destination);
                } else {
                    active.remove(&destination);
                }
            }
        }

        let finish_started = Instant::now();
        let rewritten = self
            .finish_handoffs(
                formal_plan,
                &graph,
                tables,
                &handoff_services,
                &random_state,
            )
            .await?;
        if self.config.log_transfer_steps {
            eprintln!(
                "[Bloom] finalize_ms={:.3} total_ms={:.3}",
                finish_started.elapsed().as_secs_f64() * 1000.0,
                transfer_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(rewritten)
    }

    /// Close the transfer phase and atomically construct the inputs consumed by
    /// formal execution.
    ///
    /// Sources already committed during propagation retain their handoff.
    /// Terminal single-key destinations may attach membership directly to the
    /// formal scan; all other filtered destinations are materialized exactly.
    async fn finish_handoffs(
        &self,
        formal_plan: Arc<dyn ExecutionPlan>,
        graph: &BloomGraph,
        mut tables: Vec<TableRuntime>,
        services: &HandoffServices,
        random_state: &RandomState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut retained_handoff_ids = vec![];
        let mut direct_handoff_ids = vec![];
        for (id, table) in tables.iter().enumerate() {
            if table.handoff.is_some() {
                retained_handoff_ids.push(id);
            } else if !table.pending_filters.is_empty() {
                // Match DuckDB Bloom's join-stage handoff. A destination that
                // never became a transfer source stays on its ordinary scan
                // path when every temporary structure is a single-column
                // predicate. Composite membership cannot be represented as a
                // per-column scan filter, so retain the exact materialization
                // fallback for that case.
                if table
                    .pending_filters
                    .iter()
                    .all(|filter| filter.keys.len() == 1)
                {
                    direct_handoff_ids.push(id);
                } else {
                    retained_handoff_ids.push(id);
                }
            }
        }
        let propagated_handoff_count = retained_handoff_ids
            .iter()
            .filter(|&&id| tables[id].applied_filter_count > 0)
            .count();
        if propagated_handoff_count == 0 && !direct_handoff_ids.is_empty() {
            // No destination ever became a transfer source. The work stopped
            // at ordinary build-to-probe predicates, which DataFusion's hash
            // joins already create during formal execution. Discard the
            // optimizer-time scans instead of handing duplicate predicates to
            // the same Parquet scans.
            if self.config.log_transfer_steps {
                eprintln!("[Bloom] handoff skipped: native_first_hop_coverage");
            }
            return Ok(formal_plan);
        }
        if retained_handoff_ids.is_empty() && direct_handoff_ids.is_empty() {
            return Ok(formal_plan);
        }

        // Composite destinations have exact filters already but cannot use a
        // per-column scan handoff. Collect those independent table operators
        // before the atomic plan rewrite; no formal join is executed here.
        let jobs = retained_handoff_ids
            .iter()
            .copied()
            .filter(|&id| tables[id].handoff.is_none())
            .map(|id| -> Result<_> {
                let filters = tables[id].pending_filters.clone();
                let input_row_hint = tables[id].initial_estimate.ceil() as usize;
                let required_columns = required_join_columns(graph, id)?;
                let (full_row_width, transfer_row_width) = materialization_widths(
                    services.policy,
                    tables[id].sample.as_deref(),
                    graph.tables[id].plan.schema().as_ref(),
                    &required_columns,
                );
                let table = graph.tables[id].clone();
                let locally_filtered_rows = tables[id].initial_estimate.ceil() as usize;
                let expected_rows = tables[id].estimated_rows.ceil() as usize;
                let services = services.clone();
                Ok(async move {
                    let handoff = collect_transfer_handoff(
                        HandoffRequest {
                            table: &table,
                            filters: &filters,
                            required_columns: &required_columns,
                            input_row_hint,
                            locally_filtered_rows,
                            expected_rows,
                            full_row_width,
                            transfer_row_width,
                        },
                        &services,
                    )
                    .await?;
                    Ok::<_, DataFusionError>((id, handoff, filters.len()))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for (id, handoff, applied_filters) in try_join_all(jobs).await? {
            tables[id].handoff = Some(handoff);
            tables[id].applied_filter_count = applied_filters;
        }

        if self.config.log_transfer_steps {
            let full_rows = retained_handoff_ids
                .iter()
                .filter(|&&id| {
                    matches!(
                        tables[id].handoff.as_ref(),
                        Some(TransferHandoff::FullRows(_))
                    )
                })
                .count();
            let row_locations = retained_handoff_ids
                .iter()
                .filter(|&&id| {
                    matches!(
                        tables[id].handoff.as_ref(),
                        Some(TransferHandoff::RowLocations { .. })
                    )
                })
                .count();
            eprintln!(
                "[Bloom] handoffs full_rows={} row_locations={} direct={}",
                full_rows,
                row_locations,
                direct_handoff_ids.len()
            );
        }

        let mut replacements =
            Vec::with_capacity(retained_handoff_ids.len() + direct_handoff_ids.len());
        for id in retained_handoff_ids {
            compact_handoff(
                &mut tables[id],
                random_state,
                services.context.session_config().batch_size(),
            )?;
            let Some(handoff) = tables[id].handoff.take() else {
                return internal_err!("missing Bloom materialization for table {id}");
            };
            let replacement = match handoff {
                TransferHandoff::FullRows(mut data) => {
                    let reservation = data.reservation.take().ok_or_else(|| {
                        DataFusionError::Internal(
                            "Bloom FullRows handoff lost its memory reservation".to_string(),
                        )
                    })?;
                    let collection = BloomCollection::try_new(
                        data.partitions,
                        graph.tables[id].plan.schema(),
                        data.input_row_count,
                        data.generation,
                        &format!("table-{id}"),
                        &services.context,
                        Some(reservation),
                    )?;
                    collection.into_exec(format!("table-{id}"))?
                }
                TransferHandoff::RowLocations { data, layout, .. } => {
                    // The transfer handoff is the materialized table-operator
                    // result: these locations have already passed P0's local
                    // predicates and every applied transfer predicate. The
                    // formal scan only fetches query payload for those exact
                    // positions. Re-evaluating local predicates would both
                    // violate that lifecycle and decode filter-only columns a
                    // second time.
                    let formal = strip_verified_local_filters(reset_plan_states(Arc::clone(
                        &graph.tables[id].plan,
                    ))?)?;
                    let options = services.context.session_config().options();
                    let formal = ProjectionPushdown::new().optimize(formal, options.as_ref())?;
                    layout.rewrite_formal_plan(formal, &data.partitions)?
                }
            };
            replacements.push((graph.tables[id].path.clone(), replacement));
        }

        for id in direct_handoff_ids {
            let replacement = formal_transfer_scan_plan(
                Arc::clone(&graph.tables[id].plan),
                &tables[id].pending_filters,
                services.context.as_ref(),
                services.parquet_membership_placement,
            )?;
            if self.config.log_transfer_steps {
                eprintln!(
                    "  [direct-scan] destination={} filters={}",
                    id,
                    tables[id].pending_filters.len()
                );
            }
            replacements.push((graph.tables[id].path.clone(), replacement));
        }

        // Paths come from P0, whose dynamic-filter options only change scan
        // expressions, not the physical operator tree. Apply the table
        // replacements to the separately planned native formal tree so all
        // untouched scans keep their join-owned runtime filters.
        let mut rewritten = formal_plan;
        replacements.sort_by_key(|(path, _)| std::cmp::Reverse(path.len()));
        for (path, replacement) in replacements {
            rewritten = replace_at_path(rewritten, &path, replacement)?;
        }
        Ok(rewritten)
    }
}

/// Establish each table's scheduling baseline without committing a handoff.
/// Local predicates may be sampled to estimate their reduction, but samples
/// never become formal query data or proof of emptiness.
async fn initialize_tables(
    graph: &BloomGraph,
    config: &BloomConfig,
    samples: Arc<PreparedSampleCache>,
    row_group_layouts: Arc<RowGroupLayoutCache>,
    context: Arc<TaskContext>,
) -> Result<Vec<TableRuntime>> {
    let jobs = graph.tables.iter().map(|table| {
        let samples = Arc::clone(&samples);
        let row_group_layouts = Arc::clone(&row_group_layouts);
        let task_context = Arc::clone(&context);
        async move {
            let statistics_estimate = estimated_rows(&table.plan)?;
            let base_estimate = source_rows(&table.plan)?
                .or(statistics_estimate)
                .unwrap_or(0);
            let mut sample = None;
            let initial_estimate = if contains_local_filter(&table.plan) {
                if let Some(sampled) = sample_table(
                    table,
                    config.sample_rows,
                    config.sampling_mode,
                    config.instant_parquet_row_groups,
                    &samples,
                    &row_group_layouts,
                    task_context,
                )
                .await?
                {
                    let estimate = if sampled.input_rows == 0 {
                        statistics_estimate.unwrap_or(base_estimate) as f64
                    } else if sampled.output_rows == 0 && sampled.input_rows < base_estimate {
                        // A partial sample can miss a rare local survivor. As
                        // in Bloom's estimator, zero then means one sampled-row
                        // of weight rather than a proven empty table.
                        (base_estimate as f64 / sampled.input_rows as f64).max(1.0)
                    } else {
                        base_estimate as f64 * sampled.output_rows as f64
                            / sampled.input_rows as f64
                    };
                    sample = Some(sampled.partitions);
                    estimate
                } else {
                    statistics_estimate.unwrap_or(base_estimate) as f64
                }
            } else {
                statistics_estimate.unwrap_or(base_estimate) as f64
            };
            Ok::<_, DataFusionError>(TableRuntime {
                initial_estimate,
                estimated_rows: initial_estimate,
                baseline_rows: (base_estimate as f64).max(initial_estimate),
                pending_filters: vec![],
                applied_filter_count: 0,
                sample,
                handoff: None,
            })
        }
    });
    try_join_all(jobs).await
}

/// Seed propagation using cardinality reduction alone. Storage representation
/// and row-location locality deliberately do not participate in excitation.
#[cfg(test)]
mod tests;
