use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

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
use datafusion::parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_expr::utils::{conjunction, split_conjunction};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::filter_pushdown::FilterPushdown as FilterPushdownRule;
use datafusion::physical_optimizer::projection_pushdown::ProjectionPushdown;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::execution_plan::{collect_partitioned, reset_plan_states};
use datafusion::physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion::physical_plan::filter_pushdown::{
    ChildFilterPushdownResult, ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation,
    PushedDown,
};
use datafusion::physical_plan::limit::LocalLimitExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::scalar_subquery::ScalarSubqueryExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream, with_new_children_if_necessary,
};
use futures::future::try_join_all;

use crate::collection::BloomCollection;
use crate::config::{BloomConfig, HandoffPolicy, ParquetMembershipPlacement};
use crate::filter::TransferBloomFilter;
use crate::graph::{BloomEdge, BloomGraph, BloomTable, TableId};
use crate::handoff::{
    MaterializationFacts, MaterializationStrategy, choose_materialization,
    estimated_projection_width, estimated_schema_width, row_locations_are_concentrated,
};
use crate::late_materialization::{
    PreparedRowGroupLayoutCache, RowLocationLayout, canonical_files, local_path,
    try_prepare_location_plan,
};
use crate::lineage::{LineageSnapshot, LineageTracker};
use crate::materialization::{
    compact_materialized_partition, compact_materialized_partitions, observed_handoff_widths,
    partition_physical_bytes,
};
use crate::samples::{PreparedSampleCache, PreparedSourceSample};

mod sampling;
use sampling::sample_table;
mod handoff;
use handoff::{
    collect_transfer_handoff, ensure_transfer_handoff, formal_transfer_scan_plan,
    materialization_widths, required_join_columns, strip_verified_local_filters,
};

const HASH_SEED: u64 = 0x424c_4f4f_4d44_4631;

#[derive(Debug)]
pub(crate) struct BloomTransferEngine {
    config: BloomConfig,
    samples: Arc<PreparedSampleCache>,
    row_group_layouts: Arc<PreparedRowGroupLayoutCache>,
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
    log_steps: bool,
    row_group_layouts: Arc<PreparedRowGroupLayoutCache>,
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

/// Executable membership predicate used by transfer scans and direct formal
/// handoffs. Formal Parquet scans keep this expression in Arrow so decoder
/// predicate caches are not amplified by a query-scoped membership structure.
#[derive(Debug)]
struct CascadePredicate {
    keys: Vec<Arc<dyn PhysicalExpr>>,
    filter: Arc<TransferBloomFilter>,
}

impl PartialEq for CascadePredicate {
    fn eq(&self, other: &Self) -> bool {
        self.keys == other.keys && Arc::ptr_eq(&self.filter, &other.filter)
    }
}

impl Eq for CascadePredicate {}

impl Hash for CascadePredicate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.keys.hash(state);
        (Arc::as_ptr(&self.filter) as usize).hash(state);
    }
}

impl fmt::Display for CascadePredicate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "BloomTransferMembership(")?;
        for (index, key) in self.keys.iter().enumerate() {
            if index > 0 {
                write!(formatter, ", ")?;
            }
            write!(formatter, "{key}")?;
        }
        write!(formatter, ")")
    }
}

impl PhysicalExpr for CascadePredicate {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Array(Arc::new(evaluate_membership(
            batch,
            &self.keys,
            &self.filter,
            &RandomState::with_seed(HASH_SEED),
        )?)))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.keys.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if children.len() != self.keys.len() {
            return internal_err!(
                "Bloom transfer predicate expected {} children, got {}",
                self.keys.len(),
                children.len()
            );
        }
        Ok(Arc::new(Self {
            keys: children,
            filter: Arc::clone(&self.filter),
        }))
    }

    fn fmt_sql(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

/// A transparent formal-scan boundary. Its default filter-pushdown behavior is
/// deliberately unsupported, preventing P1 from moving transfer membership
/// (or re-moving P0's local predicates) into a Parquet decoder. Safe integer
/// pruning bounds are installed below this node separately.
#[derive(Debug, Clone)]
struct BloomScanBoundaryExec {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
}

impl BloomScanBoundaryExec {
    fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let properties = Arc::clone(input.properties());
        Self { input, properties }
    }
}

impl DisplayAs for BloomScanBoundaryExec {
    fn fmt_as(
        &self,
        _display_type: DisplayFormatType,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(formatter, "BloomScanBoundaryExec")
    }
}

impl ExecutionPlan for BloomScanBoundaryExec {
    fn name(&self) -> &str {
        "BloomScanBoundaryExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!(
                "Bloom scan boundary expected one child, got {}",
                children.len()
            );
        }
        Ok(Arc::new(Self::new(children.swap_remove(0))))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.input.execute(partition, context)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        self.input.partition_statistics(partition)
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.input.supports_limit_pushdown()
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }
}

impl BloomTransferEngine {
    pub(crate) fn new(
        config: BloomConfig,
        samples: Arc<PreparedSampleCache>,
        row_group_layouts: Arc<PreparedRowGroupLayoutCache>,
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
        let Some(graph) = BloomGraph::build(&transfer_plan, &self.config)? else {
            return Ok(formal_plan);
        };
        let initialize_started = Instant::now();
        let mut tables = initialize_tables(
            &graph,
            &self.config,
            Arc::clone(&self.samples),
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
    context: Arc<TaskContext>,
) -> Result<Vec<TableRuntime>> {
    let jobs = graph.tables.iter().map(|table| {
        let samples = Arc::clone(&samples);
        let task_context = Arc::clone(&context);
        async move {
            let statistics_estimate = estimated_rows(&table.plan)?;
            let base_estimate = source_rows(&table.plan)?
                .or(statistics_estimate)
                .unwrap_or(0);
            let mut sample = None;
            let initial_estimate = if contains_local_filter(&table.plan) {
                if let Some(sampled) =
                    sample_table(table, config.sample_rows, &samples, task_context).await?
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
fn initial_active_tables(tables: &[TableRuntime], config: &BloomConfig) -> BTreeSet<TableId> {
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
fn pop_smallest_active(active: &mut BTreeSet<TableId>, tables: &[TableRuntime]) -> Option<TableId> {
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
fn collect_outgoing_edges(
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
fn compact_handoff(
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
fn install_cascade_filter(runtime: &mut TableRuntime, incoming: CascadeFilter) -> bool {
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
fn prepare_activations(
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
fn build_transfer_filters(
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
async fn estimate_destination(
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
        if let Some(sampled) =
            sample_table(table, sample_rows, samples, Arc::clone(&services.context)).await?
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

/// Evaluate transfer membership for an estimate without mutating or
/// re-owning the sampled/materialized batches.
fn count_survivors(
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
fn apply_filters(
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
fn evaluate_membership(
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

fn visit_integer_values(array: &ArrayRef, mut visitor: impl FnMut(i128)) -> bool {
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

fn evaluate_hashes(
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

fn key_signature(keys: &[Arc<dyn PhysicalExpr>]) -> Vec<usize> {
    keys.iter()
        .filter_map(|key| key.downcast_ref::<Column>().map(Column::index))
        .collect()
}

fn count_rows(partitions: &[Vec<RecordBatch>]) -> usize {
    partitions.iter().flatten().map(RecordBatch::num_rows).sum()
}

fn estimated_rows(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<usize>> {
    Ok(plan
        .partition_statistics(None)?
        .num_rows
        .get_value()
        .copied())
}

/// Recover the population entering a table-operator subtree rather than its
/// possibly filtered output statistics. This is the denominator for local
/// selectivity and never an exact formal-result claim.
fn source_rows(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<usize>> {
    let children = plan.children();
    if children.is_empty() {
        return estimated_rows(plan);
    }
    let mut total = 0usize;
    for child in children {
        let Some(rows) = source_rows(child)? else {
            return Ok(None);
        };
        total = total.saturating_add(rows);
    }
    Ok(Some(total))
}

fn contains_scalar_subquery(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<ScalarSubqueryExec>().is_some()
        || plan.children().into_iter().any(contains_scalar_subquery)
}

fn contains_local_filter(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan.downcast_ref::<FilterExec>().is_some() {
        return true;
    }
    if let Some(source) = plan.downcast_ref::<DataSourceExec>()
        && let Some(config) = source.data_source().downcast_ref::<FileScanConfig>()
        && config.file_source().filter().is_some()
    {
        return true;
    }
    plan.children().into_iter().any(contains_local_filter)
}

fn contains_filter_exec(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<FilterExec>().is_some()
        || plan.children().into_iter().any(contains_filter_exec)
}

fn contains_parquet_source(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if let Some(source) = plan.downcast_ref::<DataSourceExec>()
        && let Some(config) = source.data_source().downcast_ref::<FileScanConfig>()
        && config
            .file_source()
            .downcast_ref::<ParquetSource>()
            .is_some()
    {
        return true;
    }
    plan.children().into_iter().any(contains_parquet_source)
}

/// Replace one analyzed table-operator leaf while preserving every operator in
/// the separately planned native join tree.
fn replace_at_path(
    plan: Arc<dyn ExecutionPlan>,
    path: &[usize],
    replacement: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let Some((&index, rest)) = path.split_first() else {
        return Ok(replacement);
    };
    let mut children: Vec<_> = plan.children().into_iter().cloned().collect();
    if index >= children.len() {
        return internal_err!("Bloom plan path contains invalid child index {index}");
    }
    children[index] = replace_at_path(Arc::clone(&children[index]), rest, replacement)?;
    plan.with_new_children(children)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray, StringViewArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};

    use super::handoff::predicate_is_expensive;
    use super::sampling::localize_range;
    use super::{
        compact_materialized_partition, compact_materialized_partitions, observed_handoff_widths,
        partition_physical_bytes,
    };
    use crate::config::HandoffPolicy;
    use crate::handoff::{MaterializationFacts, MaterializationStrategy, choose_materialization};

    #[test]
    fn global_sample_ranges_are_only_localized_after_intersection() {
        assert_eq!(localize_range(&(5..10), 0, 4), None);
        assert_eq!(localize_range(&(0..4), 5, 10), None);
        assert_eq!(localize_range(&(5..10), 7, 12), Some(0..3));
        assert_eq!(localize_range(&(8..14), 7, 12), Some(1..5));
    }

    #[test]
    fn observed_buffers_keep_compressible_wide_schema_on_full_rows() {
        let mut fields = vec![Field::new("id", DataType::Int64, false)];
        fields.extend(
            (0..8).map(|index| Field::new(format!("payload_{index}"), DataType::Utf8, false)),
        );
        let schema = Arc::new(Schema::new(fields));
        let rows = 10_000;
        let ids = Arc::new(Int64Array::from_iter_values(0..rows as i64)) as ArrayRef;
        let payload = Arc::new(StringArray::from(vec!["x"; rows])) as ArrayRef;
        let mut columns = vec![ids];
        columns.extend((0..8).map(|_| Arc::clone(&payload)));
        let sample = vec![vec![
            RecordBatch::try_new(Arc::clone(&schema), columns).unwrap(),
        ]];

        let (full_row_width, transfer_row_width) =
            observed_handoff_widths(Some(&sample), schema.as_ref(), &[0]);
        let decision = choose_materialization(
            HandoffPolicy::CostBasedRowLocations,
            MaterializationFacts {
                source_rows: 300_000,
                locally_filtered_rows: 300_000,
                expected_rows: 1,
                full_row_width,
                transfer_row_width,
                has_local_filter: false,
            },
        );
        assert_eq!(decision.strategy, MaterializationStrategy::FullRows);
    }

    #[test]
    fn observed_width_does_not_charge_a_slice_for_its_backing_buffer() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("payload", DataType::Utf8, false),
        ]));
        let rows = 10_000;
        let ids = Arc::new(Int64Array::from_iter_values(0..rows as i64)) as ArrayRef;
        let payload = Arc::new(StringArray::from(vec!["0123456789abcdef"; rows])) as ArrayRef;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, payload]).unwrap();
        let sample = vec![vec![batch.slice(rows - 1, 1)]];

        let (full_row_width, transfer_row_width) =
            observed_handoff_widths(Some(&sample), schema.as_ref(), &[0]);
        assert!(full_row_width < 64, "logical width was {full_row_width}");
        assert_eq!(transfer_row_width, 20);
    }

    #[test]
    fn compact_materialization_releases_filtered_view_buffers() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "payload",
            DataType::Utf8View,
            false,
        )]));
        let values = (0..256)
            .map(|index| format!("{index:04}-{}", "x".repeat(1024)))
            .collect::<Vec<_>>();
        let payload = Arc::new(StringViewArray::from_iter_values(
            values.iter().map(String::as_str),
        )) as ArrayRef;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![payload])
            .unwrap()
            .slice(17, 1);
        let before = partition_physical_bytes(&[vec![batch.clone()]]);

        let compacted = compact_materialized_partitions(vec![vec![batch]], 65_536).unwrap();
        let after = partition_physical_bytes(&compacted);
        let payload = compacted[0][0]
            .column(0)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();

        assert_eq!(payload.value(0), values[17]);
        assert!(after * 10 < before, "before={before} after={after}");
    }

    #[test]
    fn compact_materialization_fills_batches_across_input_boundaries() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let make_batch = |start: i64| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from_iter_values(start..start + 5_000)) as ArrayRef],
            )
            .unwrap()
        };

        let compacted =
            compact_materialized_partition(vec![make_batch(0), make_batch(5_000)], 8_192).unwrap();
        assert_eq!(
            compacted
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![8_192, 1_808]
        );
        let first = compacted[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let second = compacted[1]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(first.value(8_191), 8_191);
        assert_eq!(second.value(0), 8_192);
    }

    #[test]
    fn transfer_predicate_order_classifies_strings_after_numeric_columns() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("label", DataType::Utf8, false),
        ]);
        let numeric = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", 0)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Int64(Some(42)))),
        )) as Arc<dyn PhysicalExpr>;
        let string = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("label", 1)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some("Bloom".to_string())))),
        )) as Arc<dyn PhysicalExpr>;

        assert!(!predicate_is_expensive(&numeric, &schema));
        assert!(predicate_is_expensive(&string, &schema));
    }
}
