use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Int8Array, Int16Array, Int32Array,
    Int64Array, LargeBinaryArray, LargeStringArray, StringArray, StringViewArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use datafusion::arrow::compute::{concat_batches, filter_record_batch, take};
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
    estimated_projection_width, estimated_schema_width, estimated_type_width,
    row_locations_are_concentrated,
};
use crate::late_materialization::{
    PreparedRowGroupLayoutCache, RowLocationLayout, canonical_files, local_path,
    try_prepare_location_plan,
};
use crate::lineage::{LineageSnapshot, LineageTracker};
use crate::samples::{PreparedSampleCache, PreparedSourceSample};

const HASH_SEED: u64 = 0x424c_4f4f_4d44_4631;
const MIN_COMPACTION_SAVINGS_BYTES: usize = 64 * 1024;

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
}

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
    fn full_rows(partitions: Vec<Vec<RecordBatch>>, input_row_hint: usize, filtered: bool) -> Self {
        let row_count = count_rows(&partitions);
        Self::FullRows(HandoffData {
            partitions,
            input_row_count: input_row_hint.max(row_count),
            row_count,
            generation: u64::from(filtered),
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
                let handoff = tables[source].handoff.as_ref().expect("checked above");
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
            let built = build_transfer_filters(
                tables[source].handoff.as_ref().expect("checked above"),
                &build_specs,
                &random_state,
                self.config.false_positive_rate,
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
                TransferHandoff::FullRows(data) => {
                    let collection = BloomCollection::try_new(
                        data.partitions,
                        graph.tables[id].plan.schema(),
                        data.input_row_count,
                        data.generation,
                        &format!("table-{id}"),
                        &services.context,
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
        replacements.sort_by(|(left, _), (right, _)| right.len().cmp(&left.len()));
        for (path, replacement) in replacements {
            rewritten = replace_at_path(rewritten, &path, replacement)?;
        }
        Ok(rewritten)
    }
}

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

async fn ensure_transfer_handoff(
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
            runtime.sample.as_deref(),
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

async fn collect_transfer_handoff(
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
            services.row_group_layouts.as_ref(),
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
            services.row_group_layouts.as_ref(),
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
        table,
        filters,
        input_row_hint,
        services.parquet_membership_placement,
        services.log_steps,
        Arc::clone(&services.context),
    )
    .await
}

async fn collect_full_rows_handoff(
    table: &BloomTable,
    filters: &[CascadeFilter],
    input_row_hint: usize,
    parquet_membership_placement: ParquetMembershipPlacement,
    log_transfer_steps: bool,
    context: Arc<TaskContext>,
) -> Result<TransferHandoff> {
    let prepare_started = Instant::now();
    // Preserve P0's file groups and partition count. DataFusion has already
    // sized them from `target_partitions`; creating one transfer partition per
    // physical file here would let Bloom exceed the caller's concurrency while
    // the baseline remains constrained to the configured target.
    let plan = transfer_scan_plan(
        Arc::clone(&table.plan),
        filters,
        context.as_ref(),
        parquet_membership_placement,
    )?;
    let prepare_elapsed = prepare_started.elapsed();
    let collect_started = Instant::now();
    let partitions = collect_partitioned(Arc::clone(&plan), Arc::clone(&context)).await?;
    let collect_elapsed = collect_started.elapsed();
    let physical_before = partition_physical_bytes(&partitions);
    let compact_started = Instant::now();
    let partitions =
        compact_materialized_partitions(partitions, context.session_config().batch_size())?;
    let compact_elapsed = compact_started.elapsed();
    let physical_after = partition_physical_bytes(&partitions);
    if log_transfer_steps {
        eprintln!(
            "  [materialize-phase] mode=full-rows prepare_ms={:.3} collect_ms={:.3} compact_ms={:.3}",
            prepare_elapsed.as_secs_f64() * 1000.0,
            collect_elapsed.as_secs_f64() * 1000.0,
            compact_elapsed.as_secs_f64() * 1000.0,
        );
        eprintln!(
            "  [materialize-compact] bytes_before={} bytes_after={} batches={}",
            physical_before,
            physical_after,
            partitions.iter().map(Vec::len).sum::<usize>(),
        );
        eprintln!(
            "  [materialize-metrics]\n{}",
            DisplayableExecutionPlan::with_metrics(plan.as_ref()).indent(true)
        );
    }
    Ok(TransferHandoff::full_rows(
        partitions,
        input_row_hint,
        !filters.is_empty(),
    ))
}

async fn collect_two_pass_row_location_handoff(
    table: &BloomTable,
    filters: &[CascadeFilter],
    required_columns: &[usize],
    input_row_hint: usize,
    log_transfer_steps: bool,
    row_group_layouts: &PreparedRowGroupLayoutCache,
    context: Arc<TaskContext>,
) -> Result<Option<TransferHandoff>> {
    if required_columns.is_empty() {
        return Ok(None);
    }

    let discover_prepare_started = Instant::now();
    let Some(discovery_projection) = project_local_filter_columns(table, context.as_ref())? else {
        return Ok(None);
    };
    if log_transfer_steps {
        eprintln!(
            "  [row-location-discovery-plan]\n{}",
            DisplayableExecutionPlan::new(discovery_projection.as_ref()).indent(true)
        );
    }
    let Some(discovery) =
        try_prepare_location_plan(discovery_projection, log_transfer_steps, row_group_layouts)?
    else {
        return Ok(None);
    };
    if discovery.plan.schema().fields().len() < 3 {
        if log_transfer_steps {
            eprintln!(
                "  [row-location] two-pass fallback reason=discovery width actual={} expected_at_least=3",
                discovery.plan.schema().fields().len(),
            );
        }
        return Ok(None);
    }
    let discover_prepare_elapsed = discover_prepare_started.elapsed();
    let discover_started = Instant::now();
    let local_locations =
        collect_partitioned(Arc::clone(&discovery.plan), Arc::clone(&context)).await?;
    let discover_elapsed = discover_started.elapsed();

    let selected_prepare_started = Instant::now();
    let projected = project_table_columns(table, required_columns, context.as_ref())?;
    let selected = discovery
        .layout
        .rewrite_transfer_plan(projected, &local_locations)?;
    let selected = strip_verified_local_filters(selected)?;
    let options = context.session_config().options();
    let selected = ProjectionPushdown::new().optimize(selected, options.as_ref())?;
    let remapped = remap_filters(filters, required_columns)?;
    let selected = transfer_filter_plan(selected, &remapped)?;
    let selected_prepare_elapsed = selected_prepare_started.elapsed();
    let collect_started = Instant::now();
    let partitions = collect_partitioned(Arc::clone(&selected), Arc::clone(&context)).await?;
    let locality = discovery.layout.locality(&partitions)?;
    if log_transfer_steps {
        eprintln!(
            "  [materialize-phase] mode=row-locations-two-pass discover_prepare_ms={:.3} discover_ms={:.3} selected_prepare_ms={:.3} collect_ms={:.3} local_rows={}",
            discover_prepare_elapsed.as_secs_f64() * 1000.0,
            discover_elapsed.as_secs_f64() * 1000.0,
            selected_prepare_elapsed.as_secs_f64() * 1000.0,
            collect_started.elapsed().as_secs_f64() * 1000.0,
            count_rows(&local_locations),
        );
        eprintln!(
            "  [materialize-discovery-metrics]\n{}",
            DisplayableExecutionPlan::with_metrics(discovery.plan.as_ref()).indent(true)
        );
        eprintln!(
            "  [materialize-metrics]\n{}",
            DisplayableExecutionPlan::with_metrics(selected.as_ref()).indent(true)
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
    if locality.selected_rows > 0 && !row_locations_are_concentrated(locality) {
        if log_transfer_steps {
            eprintln!("  [handoff-policy] fallback=FullRows reason=scattered_row_locations");
        }
        return Ok(None);
    }
    Ok(Some(TransferHandoff::row_locations(
        partitions,
        input_row_hint,
        !filters.is_empty(),
        required_columns.to_vec(),
        discovery.layout,
    )))
}

fn project_table_columns(
    table: &BloomTable,
    columns: &[usize],
    context: &TaskContext,
) -> Result<Arc<dyn ExecutionPlan>> {
    let plan = reset_plan_states(Arc::clone(&table.plan))?;
    project_plan_columns(plan, columns, context)
}

fn project_plan_columns(
    plan: Arc<dyn ExecutionPlan>,
    columns: &[usize],
    context: &TaskContext,
) -> Result<Arc<dyn ExecutionPlan>> {
    let schema = plan.schema();
    let expressions = columns.iter().map(|&index| {
        let field = schema.field(index);
        (
            Arc::new(Column::new(field.name(), index)) as Arc<dyn PhysicalExpr>,
            field.name().clone(),
        )
    });
    let projected = Arc::new(ProjectionExec::try_new(expressions, plan)?) as Arc<dyn ExecutionPlan>;
    let options = context.session_config().options();
    ProjectionPushdown::new().optimize(projected, options.as_ref())
}

fn project_local_filter_columns(
    table: &BloomTable,
    context: &TaskContext,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let plan = reset_plan_states(Arc::clone(&table.plan))?;
    let plan = strip_parquet_source_predicates(plan)?;
    let Some(filter) = plan.downcast_ref::<FilterExec>() else {
        return Ok(None);
    };

    let mut predicate_columns = BTreeSet::new();
    collect_expr_columns(filter.predicate(), &mut predicate_columns);
    if predicate_columns.is_empty() {
        return Ok(None);
    }
    let selected_input_columns = predicate_columns.into_iter().collect::<Vec<_>>();
    let input_schema = filter.input().schema();
    let expressions = selected_input_columns
        .iter()
        .map(|&input_index| {
            let field = input_schema.field(input_index);
            (
                Arc::new(Column::new(field.name(), input_index)) as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            )
        })
        .collect::<Vec<_>>();
    let projected_input = Arc::new(ProjectionExec::try_new(
        expressions,
        Arc::clone(filter.input()),
    )?) as Arc<dyn ExecutionPlan>;
    let column_mapping = selected_input_columns
        .into_iter()
        .enumerate()
        .map(|(new_index, old_index)| (old_index, new_index))
        .collect::<BTreeMap<_, _>>();
    let predicate = remap_expr_columns(filter.predicate(), &column_mapping)?;
    let projected = FilterExecBuilder::from(filter)
        .with_input(projected_input)
        .with_predicate(predicate)
        .apply_projection(None)?
        .build()?;
    let options = context.session_config().options();
    ProjectionPushdown::new()
        .optimize(Arc::new(projected), options.as_ref())
        .map(Some)
}

fn collect_expr_columns(expr: &Arc<dyn PhysicalExpr>, columns: &mut BTreeSet<usize>) {
    if let Some(column) = expr.downcast_ref::<Column>() {
        columns.insert(column.index());
    }
    for child in expr.children() {
        collect_expr_columns(child, columns);
    }
}

fn remap_expr_columns(
    expr: &Arc<dyn PhysicalExpr>,
    mapping: &BTreeMap<usize, usize>,
) -> Result<Arc<dyn PhysicalExpr>> {
    if let Some(column) = expr.downcast_ref::<Column>() {
        let Some(&index) = mapping.get(&column.index()) else {
            return internal_err!(
                "Bloom local predicate column {} was not retained",
                column.index()
            );
        };
        return Ok(Arc::new(Column::new(column.name(), index)));
    }
    let children = expr.children();
    if children.is_empty() {
        return Ok(Arc::clone(expr));
    }
    let children = children
        .into_iter()
        .map(|child| remap_expr_columns(child, mapping))
        .collect::<Result<Vec<_>>>()?;
    Arc::clone(expr).with_new_children(children)
}

fn row_location_transfer_plan(
    table: &BloomTable,
    filters: &[CascadeFilter],
    required_columns: &[usize],
    log_fallback: bool,
    row_group_layouts: &PreparedRowGroupLayoutCache,
    context: &TaskContext,
) -> Result<Option<RowLocationTransferPlan>> {
    if required_columns.is_empty() {
        return Ok(None);
    }
    let projected = project_table_columns(table, required_columns, context)?;
    let Some(prepared) = try_prepare_location_plan(projected, log_fallback, row_group_layouts)?
    else {
        return Ok(None);
    };
    if prepared.plan.schema().fields().len() != required_columns.len() + 2 {
        if log_fallback {
            eprintln!(
                "  [row-location] fallback reason=unexpected width actual={} expected={}",
                prepared.plan.schema().fields().len(),
                required_columns.len() + 2
            );
        }
        return Ok(None);
    }

    let remapped = remap_filters(filters, required_columns)?;
    let plan = transfer_filter_plan(prepared.plan, &remapped)?;
    Ok(Some((plan, required_columns.to_vec(), prepared.layout)))
}

fn remap_filters(
    filters: &[CascadeFilter],
    original_columns: &[usize],
) -> Result<Vec<CascadeFilter>> {
    filters
        .iter()
        .map(|filter| {
            let keys = filter
                .keys
                .iter()
                .map(|key| {
                    let column = key.downcast_ref::<Column>().ok_or_else(|| {
                        DataFusionError::Internal(
                            "Bloom row-location materialization requires column join keys"
                                .to_string(),
                        )
                    })?;
                    let position = original_columns
                        .iter()
                        .position(|index| *index == column.index())
                        .ok_or_else(|| {
                            DataFusionError::Internal(format!(
                                "Bloom row-location materialization omitted filter column {}",
                                column.index()
                            ))
                        })?;
                    Ok(Arc::new(Column::new(column.name(), position)) as Arc<dyn PhysicalExpr>)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(CascadeFilter {
                keys,
                filter: Arc::clone(&filter.filter),
                lineage: filter.lineage.clone(),
            })
        })
        .collect()
}

fn strip_verified_local_filters(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        let child = strip_verified_local_filters(Arc::clone(filter.input()))?;
        let Some(projection) = filter.projection() else {
            return Ok(child);
        };
        let output_schema = filter.schema();
        let input_schema = child.schema();
        let expressions = projection
            .iter()
            .enumerate()
            .map(|(output_index, &input_index)| {
                let input_field = input_schema.field(input_index);
                (
                    Arc::new(Column::new(input_field.name(), input_index)) as Arc<dyn PhysicalExpr>,
                    output_schema.field(output_index).name().clone(),
                )
            })
            .collect::<Vec<_>>();
        return Ok(Arc::new(ProjectionExec::try_new(expressions, child)?));
    }

    if let Some(source_exec) = plan.downcast_ref::<DataSourceExec>()
        && let Some(base) = source_exec.data_source().downcast_ref::<FileScanConfig>()
        && let Some(parquet_source) = base.file_source().downcast_ref::<ParquetSource>()
    {
        let mut clean_parquet = ParquetSource::new(parquet_source.table_schema().clone())
            .with_table_parquet_options(parquet_source.table_parquet_options().clone());
        if let Some(factory) = parquet_source.parquet_file_reader_factory() {
            clean_parquet = clean_parquet.with_parquet_file_reader_factory(Arc::clone(factory));
        }
        let mut clean_source: Arc<dyn FileSource> = Arc::new(clean_parquet);
        if let Some(projection) = parquet_source.projection() {
            let Some(projected) = clean_source.try_pushdown_projection(projection)? else {
                return internal_err!("Bloom could not recreate a selected Parquet projection");
            };
            clean_source = projected;
        }
        let clean_config = FileScanConfigBuilder::from(base.clone())
            .with_source(clean_source)
            .build();
        return Ok(DataSourceExec::from_data_source(clean_config));
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(plan);
    }
    let rewritten = children
        .into_iter()
        .map(|child| strip_verified_local_filters(Arc::clone(child)))
        .collect::<Result<Vec<_>>>()?;
    plan.with_new_children(rewritten)
}

fn strip_parquet_source_predicates(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(source_exec) = plan.downcast_ref::<DataSourceExec>()
        && let Some(base) = source_exec.data_source().downcast_ref::<FileScanConfig>()
        && let Some(parquet_source) = base.file_source().downcast_ref::<ParquetSource>()
    {
        let mut clean_parquet = ParquetSource::new(parquet_source.table_schema().clone())
            .with_table_parquet_options(parquet_source.table_parquet_options().clone());
        if let Some(factory) = parquet_source.parquet_file_reader_factory() {
            clean_parquet = clean_parquet.with_parquet_file_reader_factory(Arc::clone(factory));
        }
        let mut clean_source: Arc<dyn FileSource> = Arc::new(clean_parquet);
        if let Some(projection) = parquet_source.projection() {
            let Some(projected) = clean_source.try_pushdown_projection(projection)? else {
                return internal_err!("Bloom could not recreate a Parquet projection");
            };
            clean_source = projected;
        }
        let clean_config = FileScanConfigBuilder::from(base.clone())
            .with_source(clean_source)
            .build();
        return Ok(DataSourceExec::from_data_source(clean_config));
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(plan);
    }
    let rewritten = children
        .into_iter()
        .map(|child| strip_parquet_source_predicates(Arc::clone(child)))
        .collect::<Result<Vec<_>>>()?;
    plan.with_new_children(rewritten)
}

fn transfer_filter_plan(
    plan: Arc<dyn ExecutionPlan>,
    filters: &[CascadeFilter],
) -> Result<Arc<dyn ExecutionPlan>> {
    if filters.is_empty() {
        return Ok(plan);
    }
    let predicates = cascade_predicates(filters, plan.schema().as_ref())?;
    sequential_filter_plan(plan, predicates)
}

fn sequential_filter_plan(
    mut plan: Arc<dyn ExecutionPlan>,
    predicates: impl IntoIterator<Item = Arc<dyn PhysicalExpr>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    for predicate in predicates {
        plan = Arc::new(FilterExec::try_new(predicate, plan)?) as Arc<dyn ExecutionPlan>;
    }
    Ok(plan)
}

fn cascade_predicates(
    filters: &[CascadeFilter],
    _schema: &Schema,
) -> Result<Vec<Arc<dyn PhysicalExpr>>> {
    let mut predicates = Vec::with_capacity(filters.len());
    for cascade in filters {
        // Dense integer membership already performs its own bounds check.
        // Emitting separate >= and <= expressions at an Arrow FilterExec made
        // every fact row traverse three boolean kernels for the same key,
        // without gaining Parquet statistics pruning at this scan boundary.
        predicates.push(Arc::new(CascadePredicate {
            keys: cascade.keys.clone(),
            filter: Arc::clone(&cascade.filter),
        }) as Arc<dyn PhysicalExpr>);
    }
    Ok(predicates)
}

fn required_join_columns(graph: &BloomGraph, table: TableId) -> Result<Vec<usize>> {
    let mut columns = BTreeSet::new();
    for edge in &graph.edges {
        let keys = if edge.left == table {
            Some(&edge.left_keys)
        } else if edge.right == table {
            Some(&edge.right_keys)
        } else {
            None
        };
        for key in keys.into_iter().flatten() {
            let column = key.downcast_ref::<Column>().ok_or_else(|| {
                DataFusionError::Internal(
                    "Bloom graph contains a non-column table join key".to_string(),
                )
            })?;
            columns.insert(column.index());
        }
    }
    Ok(columns.into_iter().collect())
}

fn observed_handoff_widths(
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
            let values = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 array type");
            (values.len() + 1).saturating_mul(size_of::<i32>())
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    .map(|index| values.value(index).len())
                    .sum::<usize>()
        }
        DataType::LargeUtf8 => {
            let values = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("LargeUtf8 array type");
            (values.len() + 1).saturating_mul(size_of::<i64>())
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    .map(|index| values.value(index).len())
                    .sum::<usize>()
        }
        DataType::Utf8View => {
            let values = array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .expect("Utf8View array type");
            values.len().saturating_mul(16)
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    // Up to twelve bytes are stored inline in the view.
                    .map(|index| values.value(index).len().saturating_sub(12))
                    .sum::<usize>()
        }
        DataType::Binary => {
            let values = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("Binary array type");
            (values.len() + 1).saturating_mul(size_of::<i32>())
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    .map(|index| values.value(index).len())
                    .sum::<usize>()
        }
        DataType::LargeBinary => {
            let values = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .expect("LargeBinary array type");
            (values.len() + 1).saturating_mul(size_of::<i64>())
                + (0..values.len())
                    .filter(|&index| values.is_valid(index))
                    .map(|index| values.value(index).len())
                    .sum::<usize>()
        }
        DataType::BinaryView => {
            let values = array
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .expect("BinaryView array type");
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

/// Convert reader-produced batches into collection-owned Arrow buffers.
///
/// Parquet row filtering can return a handful of logical rows whose arrays
/// still reference megabytes of decoded page buffers. Bloom retains these
/// batches across transfer and formal execution, so keeping those references
/// turns a selective materialization into a large hidden memory reservation.
/// Coalescing small batches mirrors DuckDB's compact ColumnDataCollection; a
/// single amplified batch is copied only when doing so saves meaningful space.
fn compact_materialized_partitions(
    partitions: Vec<Vec<RecordBatch>>,
    target_batch_rows: usize,
) -> Result<Vec<Vec<RecordBatch>>> {
    partitions
        .into_iter()
        .map(|partition| compact_materialized_partition(partition, target_batch_rows.max(1)))
        .collect()
}

fn compact_materialized_partition(
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
            // Byte-view `take` preserves the complete data-buffer list. A
            // materialized view with several small backing buffers can
            // therefore make every high-fanout join output clone that list,
            // even when its total byte size is not amplified enough to trip
            // the ordinary copy threshold. FullRows is an owning handoff, so
            // canonicalize every view into one collection-owned buffer once.
            // This mirrors appending strings into DuckDB's materialized
            // collection and prevents buffer-list growth across formal joins.
            match array.data_type() {
                DataType::Utf8View => {
                    changed = true;
                    return Ok(Arc::new(
                        array
                            .as_any()
                            .downcast_ref::<StringViewArray>()
                            .expect("Utf8View array type")
                            .gc(),
                    ) as ArrayRef);
                }
                DataType::BinaryView => {
                    changed = true;
                    return Ok(Arc::new(
                        array
                            .as_any()
                            .downcast_ref::<BinaryViewArray>()
                            .expect("BinaryView array type")
                            .gc(),
                    ) as ArrayRef);
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

fn partition_physical_bytes(partitions: &[Vec<RecordBatch>]) -> usize {
    partitions
        .iter()
        .flatten()
        .flat_map(RecordBatch::columns)
        .map(|array| array.get_array_memory_size())
        .sum()
}

fn materialization_widths(
    policy: HandoffPolicy,
    sample: Option<&[Vec<RecordBatch>]>,
    schema: &Schema,
    required_columns: &[usize],
) -> (usize, usize) {
    match policy {
        HandoffPolicy::FullRows => (
            estimated_schema_width(schema),
            estimated_projection_width(schema, required_columns),
        ),
        HandoffPolicy::CostBasedRowLocations => {
            observed_handoff_widths(sample, schema, required_columns)
        }
    }
}

fn predicate_is_expensive(predicate: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    if contains_cascade_predicate(predicate) {
        return false;
    }
    if let Some(column) = predicate.downcast_ref::<Column>() {
        return schema
            .fields()
            .get(column.index())
            .is_some_and(|field| expensive_filter_type(field.data_type()));
    }
    predicate
        .children()
        .iter()
        .any(|child| predicate_is_expensive(child, schema))
}

fn expensive_filter_type(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => true,
        DataType::Dictionary(_, value_type) => expensive_filter_type(value_type),
        _ => false,
    }
}

fn transfer_scan_plan(
    plan: Arc<dyn ExecutionPlan>,
    filters: &[CascadeFilter],
    context: &TaskContext,
    parquet_membership_placement: ParquetMembershipPlacement,
) -> Result<Arc<dyn ExecutionPlan>> {
    transfer_scan_plan_impl(plan, filters, context, false, parquet_membership_placement)
}

fn formal_transfer_scan_plan(
    plan: Arc<dyn ExecutionPlan>,
    filters: &[CascadeFilter],
    context: &TaskContext,
    parquet_membership_placement: ParquetMembershipPlacement,
) -> Result<Arc<dyn ExecutionPlan>> {
    transfer_scan_plan_impl(plan, filters, context, true, parquet_membership_placement)
}

fn transfer_scan_plan_impl(
    plan: Arc<dyn ExecutionPlan>,
    filters: &[CascadeFilter],
    context: &TaskContext,
    formal_boundary: bool,
    parquet_membership_placement: ParquetMembershipPlacement,
) -> Result<Arc<dyn ExecutionPlan>> {
    let plan = reset_plan_states(plan)?;
    if filters.is_empty() {
        return Ok(plan);
    }
    let has_parquet_source = contains_parquet_source(&plan);

    let predicates = cascade_predicates(filters, plan.schema().as_ref())?;
    let mut options = context.session_config().options().as_ref().clone();
    options.execution.parquet.pushdown_filters =
        parquet_membership_placement == ParquetMembershipPlacement::Reader;
    options.execution.parquet.reorder_filters = false;

    // P0 already contains the query's local FilterExec nodes and Parquet
    // pruning predicates. Running DataFusion's complete FilterPushdown rule a
    // second time would inject those local predicates into the source again.
    // Push only the newly built transfer predicates through the existing tree;
    // self-provided filters stay where P0 put them. This is the physical-plan
    // equivalent of DuckDB Bloom attaching only new table filters to a scan.
    let pushed = push_incremental_filters(&plan, predicates.clone(), &options, formal_boundary)?;
    let rewritten = pushed.updated_node.unwrap_or(plan);
    let residual = predicates
        .into_iter()
        .zip(pushed.filters)
        .filter_map(|(predicate, result)| matches!(result, PushedDown::No).then_some(predicate))
        .collect::<Vec<_>>();
    let rewritten = if residual.is_empty() {
        rewritten
    } else {
        sequential_filter_plan(rewritten, residual)?
    };
    // The boundary keeps post-scan membership outside the reader, while the
    // optimizer can still move independent integer pruning hints into the
    // source. Reader placement has no boundary and pushes membership too.
    // Only Parquet needs a second, general optimizer pass to convert the
    // staged filters into a reader RowFilter. On other bounded sources that
    // pass would merge Bloom membership back into P0's local FilterExec,
    // forcing expensive local expressions to run over every input row. Keep
    // the two FilterExec stages emitted above: membership first, P0 predicate
    // second over only the survivors.
    let rewritten = if has_parquet_source {
        FilterPushdownRule::new().optimize(rewritten, &options)?
    } else {
        rewritten
    };
    let rewritten = if has_parquet_source && options.execution.parquet.pushdown_filters {
        prioritize_parquet_membership(rewritten)?
    } else {
        rewritten
    };
    ProjectionPushdown::new().optimize(rewritten, &options)
}

/// Put Bloom membership before P0-owned local predicates in a Parquet
/// `RowFilter`. DataFusion's optional generic reordering only considers each
/// predicate's compressed input bytes, not the cardinality already measured by
/// Bloom. A narrow integer key can therefore be placed after an expensive
/// string predicate even when it rejects nearly every row.
fn prioritize_parquet_membership(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(source_exec) = plan.downcast_ref::<DataSourceExec>()
        && let Some(base) = source_exec.data_source().downcast_ref::<FileScanConfig>()
        && let Some(parquet_source) = base.file_source().downcast_ref::<ParquetSource>()
        && let Some(predicate) = parquet_source.filter()
    {
        let mut conjuncts = Vec::new();
        for predicate in split_conjunction(&predicate).into_iter().cloned() {
            if !conjuncts
                .iter()
                .any(|existing: &Arc<dyn PhysicalExpr>| existing.eq(&predicate))
            {
                conjuncts.push(predicate);
            }
        }
        let source_schema = source_exec.schema();
        conjuncts.sort_by_key(|predicate| {
            if contains_cascade_predicate(predicate) {
                0
            } else if predicate_is_expensive(predicate, source_schema.as_ref()) {
                2
            } else {
                1
            }
        });
        let predicate = conjunction(conjuncts);
        let reordered = parquet_source
            .with_predicate(predicate)
            .with_pushdown_filters(true)
            .with_reorder_filters(false);
        let config = FileScanConfigBuilder::from(base.clone())
            .with_source(Arc::new(reordered))
            .build();
        return Ok(DataSourceExec::from_data_source(config));
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(plan);
    }
    let children = children
        .into_iter()
        .map(|child| prioritize_parquet_membership(Arc::clone(child)))
        .collect::<Result<Vec<_>>>()?;
    plan.with_new_children(children)
}

fn contains_cascade_predicate(predicate: &Arc<dyn PhysicalExpr>) -> bool {
    predicate.downcast_ref::<CascadePredicate>().is_some()
        || predicate
            .children()
            .iter()
            .any(|child| contains_cascade_predicate(child))
}

fn integer_pruning_predicates(
    predicates: &[Arc<dyn PhysicalExpr>],
    schema: &Schema,
    exact_only: bool,
) -> Result<Vec<Arc<dyn PhysicalExpr>>> {
    let mut bounds = vec![];
    for predicate in predicates {
        let Some(cascade) = predicate.downcast_ref::<CascadePredicate>() else {
            continue;
        };
        let [key] = cascade.keys.as_slice() else {
            continue;
        };
        let Some((minimum, maximum)) = cascade.filter.integer_bounds() else {
            continue;
        };
        if exact_only && minimum != maximum {
            continue;
        }
        let data_type = key.data_type(schema)?;
        let Some(minimum) = integer_scalar(minimum, &data_type) else {
            continue;
        };
        let Some(maximum) = integer_scalar(maximum, &data_type) else {
            continue;
        };
        if minimum == maximum {
            bounds.push(Arc::new(BinaryExpr::new(
                Arc::clone(key),
                Operator::Eq,
                Arc::new(Literal::new(minimum)),
            )) as Arc<dyn PhysicalExpr>);
        } else {
            bounds.push(Arc::new(BinaryExpr::new(
                Arc::clone(key),
                Operator::GtEq,
                Arc::new(Literal::new(minimum)),
            )) as Arc<dyn PhysicalExpr>);
            bounds.push(Arc::new(BinaryExpr::new(
                Arc::clone(key),
                Operator::LtEq,
                Arc::new(Literal::new(maximum)),
            )) as Arc<dyn PhysicalExpr>);
        }
    }
    Ok(bounds)
}

fn integer_scalar(value: i128, data_type: &DataType) -> Option<ScalarValue> {
    match data_type {
        DataType::Int8 => i8::try_from(value)
            .ok()
            .map(|value| ScalarValue::Int8(Some(value))),
        DataType::Int16 => i16::try_from(value)
            .ok()
            .map(|value| ScalarValue::Int16(Some(value))),
        DataType::Int32 => i32::try_from(value)
            .ok()
            .map(|value| ScalarValue::Int32(Some(value))),
        DataType::Int64 => i64::try_from(value)
            .ok()
            .map(|value| ScalarValue::Int64(Some(value))),
        DataType::UInt8 => u8::try_from(value)
            .ok()
            .map(|value| ScalarValue::UInt8(Some(value))),
        DataType::UInt16 => u16::try_from(value)
            .ok()
            .map(|value| ScalarValue::UInt16(Some(value))),
        DataType::UInt32 => u32::try_from(value)
            .ok()
            .map(|value| ScalarValue::UInt32(Some(value))),
        DataType::UInt64 => u64::try_from(value)
            .ok()
            .map(|value| ScalarValue::UInt64(Some(value))),
        _ => None,
    }
}

/// Push predicates supplied by Bloom through an already optimized P0 plan
/// without re-pushing predicates owned by existing operators.
fn push_incremental_filters(
    node: &Arc<dyn ExecutionPlan>,
    parent_predicates: Vec<Arc<dyn PhysicalExpr>>,
    config: &ConfigOptions,
    formal_boundary: bool,
) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
    // This is the only storage boundary at which placement is decided. Reader
    // mode lets DataFusion turn these nodes into a Parquet RowFilter; PostScan
    // mode protects them with BloomScanBoundaryExec and evaluates them over
    // decoded Arrow batches. Existing P0 predicates remain untouched in both
    // modes. Predicates can still be routed through projections and grouping
    // aggregates before reaching this boundary.
    if node.downcast_ref::<DataSourceExec>().is_some() && !parent_predicates.is_empty() {
        if config.execution.parquet.pushdown_filters {
            // Put Bloom's predicates immediately above the source, below P0's
            // local FilterExec nodes. DataFusion preserves this conjunction
            // order when building a Parquet RowFilter, so cheap membership is
            // evaluated before filter-only string columns are decoded.
            // Exact integer bounds are a separate storage hint: membership is
            // the semantic row filter, while equality can prune Parquet pages
            // and row groups before either predicate is evaluated.
            let pruning_predicates = integer_pruning_predicates(
                &parent_predicates,
                node.schema().as_ref(),
                !formal_boundary,
            )?;
            let bounded = sequential_filter_plan(Arc::clone(node), pruning_predicates)?;
            let filter = sequential_filter_plan(bounded, parent_predicates.iter().cloned())?;
            return Ok(FilterPushdownPropagation {
                filters: vec![PushedDown::Yes; parent_predicates.len()],
                updated_node: Some(filter),
            });
        }
        let pruning_predicates = integer_pruning_predicates(
            &parent_predicates,
            node.schema().as_ref(),
            !formal_boundary,
        )?;
        let bounded = sequential_filter_plan(Arc::clone(node), pruning_predicates)?;
        let input = Arc::new(BloomScanBoundaryExec::new(bounded)) as Arc<dyn ExecutionPlan>;
        let filter = sequential_filter_plan(input, parent_predicates.iter().cloned())?;
        return Ok(FilterPushdownPropagation {
            filters: vec![PushedDown::Yes; parent_predicates.len()],
            updated_node: Some(filter),
        });
    }

    let children = node.children();
    let description = node.gather_filters_for_pushdown(
        FilterPushdownPhase::Pre,
        parent_predicates.clone(),
        config,
    )?;
    let routed_parent_filters = description.parent_filters();
    let existing_self_filters = description.self_filters();
    if routed_parent_filters.len() != children.len()
        || existing_self_filters.len() != children.len()
    {
        return internal_err!(
            "incremental Bloom filter pushdown received an invalid description from {}",
            node.name()
        );
    }

    let mut parent_results = vec![vec![PushedDown::No; children.len()]; parent_predicates.len()];
    let mut self_results = Vec::with_capacity(children.len());
    let mut new_children = Vec::with_capacity(children.len());

    for (child_index, child) in children.iter().enumerate() {
        let routed = &routed_parent_filters[child_index];
        if routed.len() != parent_predicates.len() {
            return internal_err!(
                "incremental Bloom filter pushdown received {} parent filters from {}, expected {}",
                routed.len(),
                node.name(),
                parent_predicates.len()
            );
        }

        let mut child_predicates = Vec::new();
        let mut parent_indices = Vec::new();
        for (parent_index, predicate) in routed.iter().enumerate() {
            if matches!(predicate.discriminant, PushedDown::Yes) {
                parent_indices.push(parent_index);
                child_predicates.push(Arc::clone(&predicate.predicate));
            }
        }

        let child_result =
            push_incremental_filters(child, child_predicates, config, formal_boundary)?;
        if child_result.filters.len() != parent_indices.len() {
            return internal_err!(
                "incremental Bloom filter pushdown received {} child results from {}, expected {}",
                child_result.filters.len(),
                child.name(),
                parent_indices.len()
            );
        }
        for (parent_index, result) in parent_indices.into_iter().zip(child_result.filters) {
            parent_results[parent_index][child_index] = result;
        }
        new_children.push(
            child_result
                .updated_node
                .unwrap_or_else(|| Arc::clone(child)),
        );

        // Existing FilterExec/dynamic filters belong to P0. Reporting them as
        // unsupported keeps their current nodes intact while the parent Bloom
        // predicates continue independently.
        self_results.push(
            existing_self_filters[child_index]
                .iter()
                .cloned()
                .map(|predicate| PushedDown::No.wrap_expression(predicate))
                .collect(),
        );
    }

    let updated_node = with_new_children_if_necessary(Arc::clone(node), new_children)?;
    let mut result = updated_node.handle_child_pushdown_result(
        FilterPushdownPhase::Pre,
        ChildPushdownResult {
            parent_filters: parent_predicates
                .into_iter()
                .enumerate()
                .map(|(index, filter)| ChildFilterPushdownResult {
                    filter,
                    child_results: parent_results[index].clone(),
                })
                .collect(),
            self_filters: self_results,
        },
        config,
    )?;
    if result.updated_node.is_none() && !Arc::ptr_eq(&updated_node, node) {
        result.updated_node = Some(updated_node);
    }
    Ok(result)
}

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
    data.partitions = apply_filters(
        std::mem::take(&mut data.partitions),
        &filters,
        random_state,
        target_batch_rows,
    )?;
    data.row_count = count_rows(&data.partitions);
    if data.row_count < original_rows {
        data.generation += 1;
    }
    runtime.applied_filter_count = runtime.pending_filters.len();
    Ok(())
}

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

    let mut filters = bounds
        .iter()
        .enumerate()
        .map(|(index, bounds)| {
            if integral[index]
                && let Some((minimum, maximum)) = bounds
                && let Some(max_bits) = dense_integer_bits(*minimum, *maximum, source.row_count())
                && let Some(filter) =
                    TransferBloomFilter::dense_integer(*minimum, *maximum, max_bits)
            {
                return filter;
            }
            TransferBloomFilter::with_capacity(source.row_count(), false_positive_rate)
        })
        .collect::<Vec<_>>();

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

fn dense_integer_bits(minimum: i128, maximum: i128, rows: usize) -> Option<usize> {
    let range = maximum.checked_sub(minimum)?;
    let span = range.checked_add(1)?;
    let density_limit = (rows as i128).saturating_mul(128);
    if range > 8_000_000 && range > density_limit {
        return None;
    }
    usize::try_from(span).ok()
}

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
    let sample = runtime.sample.as_ref().expect("sample initialized");
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

#[derive(Debug)]
struct SampledTable {
    partitions: Vec<Vec<RecordBatch>>,
    input_rows: usize,
    output_rows: usize,
}

async fn sample_table(
    table: &BloomTable,
    target_rows: usize,
    samples: &PreparedSampleCache,
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
    } else if let Some(prepared) =
        prepared_parquet_sample(source_plan, target_rows, samples, Arc::clone(&context)).await?
    {
        prepared
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

async fn prepared_parquet_sample(
    source_plan: &Arc<dyn ExecutionPlan>,
    target_rows: usize,
    samples: &PreparedSampleCache,
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
    let prepared = if let Some(sample) = samples.get(&key)? {
        sample
    } else {
        let files = scattered_sample_files(base, target_rows)?;
        if files.is_empty() {
            return Ok(None);
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
        let sample = PreparedSourceSample {
            input_rows: count_rows(&partitions),
            partitions,
            schema,
        };
        samples.insert(key, sample.clone())?;
        sample
    };

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

fn prepared_parquet_sample_key(config: &FileScanConfig, target_rows: usize) -> String {
    let mut files = config
        .file_groups
        .iter()
        .flat_map(FileGroup::iter)
        .map(|file| {
            format!(
                "{}:{}:{:?}:{:?}:{}:{}",
                file.object_meta.location,
                file.object_meta.size,
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

fn scattered_sample_files(
    config: &FileScanConfig,
    target_rows: usize,
) -> Result<Vec<datafusion_datasource::PartitionedFile>> {
    let files = canonical_files(config)?;
    let mut row_groups = Vec::with_capacity(files.len());
    let mut total_rows = 0usize;
    for file in &files {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(local_path(file))?)?;
        let groups = reader
            .metadata()
            .row_groups()
            .iter()
            .map(|group| group.num_rows() as usize)
            .collect::<Vec<_>>();
        total_rows = total_rows.saturating_add(groups.iter().sum::<usize>());
        row_groups.push(groups);
    }
    if total_rows == 0 {
        return Ok(files);
    }
    if total_rows <= target_rows {
        return Ok(files);
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
    Ok(output)
}

fn localize_range(
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

    use super::{
        compact_materialized_partition, compact_materialized_partitions, localize_range,
        observed_handoff_widths, partition_physical_bytes, predicate_is_expensive,
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
