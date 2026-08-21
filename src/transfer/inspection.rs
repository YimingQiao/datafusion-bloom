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
