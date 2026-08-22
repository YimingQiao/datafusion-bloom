//! Physical transfer predicate and the optional post-scan pushdown boundary.

use super::*;
use datafusion::physical_plan::statistics::{ChildStats, StatisticsArgs};

/// Executable membership predicate used by transfer scans and direct formal
/// handoffs. Formal Parquet scans keep this expression in Arrow so decoder
/// predicate caches are not amplified by a query-scoped membership structure.
#[derive(Debug)]
pub(super) struct CascadePredicate {
    pub(super) keys: Vec<Arc<dyn PhysicalExpr>>,
    pub(super) filter: Arc<TransferBloomFilter>,
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
pub(super) struct BloomScanBoundaryExec {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
}

impl BloomScanBoundaryExec {
    pub(super) fn new(input: Arc<dyn ExecutionPlan>) -> Self {
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
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

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

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        let [input_stats] = input_stats else {
            return internal_err!(
                "Bloom scan boundary expected one child statistic, got {}",
                input_stats.len()
            );
        };
        Ok(Arc::clone(input_stats))
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.input.supports_limit_pushdown()
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }
}
