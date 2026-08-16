use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use datafusion::common::Result;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::source::DataSourceExec;
use datafusion::logical_expr::JoinType;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::execution_plan::Boundedness;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::projection::ProjectionExec;

use crate::collection::BloomCollectionSource;
use crate::config::BloomConfig;

pub(crate) type TableId = usize;

#[derive(Debug, Clone)]
pub(crate) struct BloomTable {
    pub(crate) path: Vec<usize>,
    pub(crate) plan: Arc<dyn ExecutionPlan>,
    pub(crate) repeatable: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct BloomEdge {
    pub(crate) left: TableId,
    pub(crate) right: TableId,
    pub(crate) left_keys: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) right_keys: Vec<Arc<dyn PhysicalExpr>>,
}

#[derive(Debug)]
pub(crate) struct BloomGraph {
    pub(crate) tables: Vec<BloomTable>,
    pub(crate) edges: Vec<BloomEdge>,
}

impl BloomGraph {
    pub(crate) fn build(
        plan: &Arc<dyn ExecutionPlan>,
        config: &BloomConfig,
    ) -> Result<Option<Self>> {
        if !contains_hash_join(plan) {
            return Ok(None);
        }

        let mut builder = GraphBuilder::new(config);
        let mut path = vec![];
        builder.analyze(plan, &mut path)?;
        if builder.edges.is_empty() {
            return Ok(None);
        }

        Ok(Some(Self {
            tables: builder.tables,
            edges: builder.edges,
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Origin {
    table: TableId,
    column: usize,
}

struct GraphBuilder<'a> {
    config: &'a BloomConfig,
    tables: Vec<BloomTable>,
    table_by_path: HashMap<Vec<usize>, TableId>,
    edges: Vec<BloomEdge>,
}

impl<'a> GraphBuilder<'a> {
    fn new(config: &'a BloomConfig) -> Self {
        Self {
            config,
            tables: vec![],
            table_by_path: HashMap::new(),
            edges: vec![],
        }
    }

    /// Analyze `plan` and return the base-table origin of each output column.
    fn analyze(
        &mut self,
        plan: &Arc<dyn ExecutionPlan>,
        path: &mut Vec<usize>,
    ) -> Result<Vec<Option<Origin>>> {
        if !contains_any_join(plan) {
            return Ok(self.register_table(plan, path));
        }

        if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
            return self.analyze_hash_join(join, path);
        }

        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            path.push(0);
            let input_lineage = self.analyze(projection.input(), path)?;
            path.pop();

            return Ok(projection
                .expr()
                .iter()
                .map(|projection_expr| {
                    projection_expr
                        .expr
                        .downcast_ref::<Column>()
                        .and_then(|column| input_lineage.get(column.index()).copied().flatten())
                })
                .collect());
        }

        let children = plan.children();
        if children.len() == 1 && is_identity_wrapper(plan.name()) {
            path.push(0);
            let lineage = self.analyze(children[0], path)?;
            path.pop();
            if lineage.len() == plan.schema().fields().len() {
                return Ok(lineage);
            }
        } else {
            for (index, child) in children.into_iter().enumerate() {
                path.push(index);
                self.analyze(child, path)?;
                path.pop();
            }
        }

        Ok(vec![None; plan.schema().fields().len()])
    }

    fn analyze_hash_join(
        &mut self,
        join: &HashJoinExec,
        path: &mut Vec<usize>,
    ) -> Result<Vec<Option<Origin>>> {
        path.push(0);
        let left_lineage = self.analyze(join.left(), path)?;
        path.pop();

        path.push(1);
        let right_lineage = self.analyze(join.right(), path)?;
        path.pop();

        if join.join_type() != &JoinType::Inner {
            return Ok(vec![None; join.schema().fields().len()]);
        }

        // A physical join may contain key pairs from different originating
        // table pairs. Group pairs so composite keys remain composite.
        let mut grouped: BTreeMap<(TableId, TableId), (Vec<_>, Vec<_>)> = BTreeMap::new();
        for (left_expr, right_expr) in join.on() {
            let Some(left_origin) = resolve_column(left_expr, &left_lineage) else {
                continue;
            };
            let Some(right_origin) = resolve_column(right_expr, &right_lineage) else {
                continue;
            };
            if left_origin.table == right_origin.table {
                continue;
            }

            let left_table = &self.tables[left_origin.table];
            let right_table = &self.tables[right_origin.table];
            let left_name = left_table
                .plan
                .schema()
                .field(left_origin.column)
                .name()
                .clone();
            let right_name = right_table
                .plan
                .schema()
                .field(right_origin.column)
                .name()
                .clone();
            let pair = grouped
                .entry((left_origin.table, right_origin.table))
                .or_default();
            pair.0.push(
                Arc::new(Column::new(&left_name, left_origin.column)) as Arc<dyn PhysicalExpr>
            );
            pair.1
                .push(Arc::new(Column::new(&right_name, right_origin.column))
                    as Arc<dyn PhysicalExpr>);
        }

        self.edges.extend(
            grouped
                .into_iter()
                .map(|((left, right), (left_keys, right_keys))| BloomEdge {
                    left,
                    right,
                    left_keys,
                    right_keys,
                }),
        );

        let mut output_lineage = left_lineage;
        output_lineage.extend(right_lineage);
        if let Some(projection) = join.projection.as_deref() {
            output_lineage = projection
                .iter()
                .map(|&index| output_lineage.get(index).copied().flatten())
                .collect();
        }
        if output_lineage.len() != join.schema().fields().len() {
            output_lineage.resize(join.schema().fields().len(), None);
        }
        Ok(output_lineage)
    }

    fn register_table(
        &mut self,
        plan: &Arc<dyn ExecutionPlan>,
        path: &[usize],
    ) -> Vec<Option<Origin>> {
        if plan.properties().boundedness != Boundedness::Bounded {
            return vec![None; plan.schema().fields().len()];
        }
        if self.config.memory_sources_only && !is_repeatable_memory_subtree(plan) {
            return vec![None; plan.schema().fields().len()];
        }

        let id = if let Some(id) = self.table_by_path.get(path) {
            *id
        } else {
            let id = self.tables.len();
            self.tables.push(BloomTable {
                path: path.to_vec(),
                plan: Arc::clone(plan),
                repeatable: is_repeatable_memory_subtree(plan),
            });
            self.table_by_path.insert(path.to_vec(), id);
            id
        };

        (0..plan.schema().fields().len())
            .map(|column| Some(Origin { table: id, column }))
            .collect()
    }
}

fn resolve_column(expr: &Arc<dyn PhysicalExpr>, lineage: &[Option<Origin>]) -> Option<Origin> {
    let column = expr.downcast_ref::<Column>()?;
    lineage.get(column.index()).copied().flatten()
}

fn contains_hash_join(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<HashJoinExec>().is_some()
        || plan.children().into_iter().any(contains_hash_join)
}

fn contains_any_join(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.name().contains("JoinExec") || plan.children().into_iter().any(contains_any_join)
}

fn is_identity_wrapper(name: &str) -> bool {
    matches!(
        name,
        "BufferExec"
            | "CoalesceBatchesExec"
            | "CoalescePartitionsExec"
            | "CooperativeExec"
            | "FilterExec"
            | "GlobalLimitExec"
            | "LocalLimitExec"
            | "OutputRequirementExec"
            | "RepartitionExec"
            | "SortExec"
            | "SortPreservingMergeExec"
    )
}

fn is_repeatable_memory_subtree(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
        return source
            .data_source()
            .downcast_ref::<MemorySourceConfig>()
            .is_some()
            || source
                .data_source()
                .downcast_ref::<BloomCollectionSource>()
                .is_some();
    }

    let children = plan.children();
    if children.is_empty() {
        return plan.name() == "EmptyExec";
    }
    children.into_iter().all(is_repeatable_memory_subtree)
}
