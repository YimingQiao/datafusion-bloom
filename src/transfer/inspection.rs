//! Read-only physical-plan inspection used by transfer admission and rewrite.

use super::*;

pub(super) fn estimated_rows(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<usize>> {
    Ok(plan
        .partition_statistics(None)?
        .num_rows
        .get_value()
        .copied())
}

/// Recover the population entering a table-operator subtree rather than its
/// possibly filtered output statistics. This is the denominator for local
/// selectivity and never an exact formal-result claim.
pub(super) fn source_rows(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<usize>> {
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct NativeJoinFilterCoverage {
    pub(super) join_count: usize,
    pub(super) collect_left: usize,
}

/// Count the inner hash-join boundaries for which DataFusion's formal plan
/// already supplies a build-to-probe dynamic filter.
pub(super) fn native_join_filter_coverage(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<NativeJoinFilterCoverage> {
    let mut coverage = NativeJoinFilterCoverage::default();
    if !collect_join_filter_coverage(plan, &mut coverage) || coverage.join_count == 0 {
        return None;
    }
    Some(coverage)
}

fn collect_join_filter_coverage(
    plan: &Arc<dyn ExecutionPlan>,
    coverage: &mut NativeJoinFilterCoverage,
) -> bool {
    if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
        if join.join_type() != &datafusion::logical_expr::JoinType::Inner {
            return false;
        }
        coverage.join_count += 1;
        if join.partition_mode() == &PartitionMode::CollectLeft {
            coverage.collect_left += 1;
        }
    } else if plan.downcast_ref::<CrossJoinExec>().is_some()
        || plan.downcast_ref::<NestedLoopJoinExec>().is_some()
        || plan.downcast_ref::<PiecewiseMergeJoinExec>().is_some()
        || plan.downcast_ref::<SortMergeJoinExec>().is_some()
        || plan.downcast_ref::<SymmetricHashJoinExec>().is_some()
    {
        // A coverage ratio over only the hash joins is not representative when
        // the formal plan contains another join implementation.
        return false;
    }
    plan.children()
        .into_iter()
        .all(|child| collect_join_filter_coverage(child, coverage))
}

pub(super) fn contains_scalar_subquery(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<ScalarSubqueryExec>().is_some()
        || plan.children().into_iter().any(contains_scalar_subquery)
}

pub(super) fn contains_local_filter(plan: &Arc<dyn ExecutionPlan>) -> bool {
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

pub(super) fn contains_filter_exec(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<FilterExec>().is_some()
        || plan.children().into_iter().any(contains_filter_exec)
}

pub(super) fn contains_parquet_source(plan: &Arc<dyn ExecutionPlan>) -> bool {
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
pub(super) fn replace_at_path(
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
