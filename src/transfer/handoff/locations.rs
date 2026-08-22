//! Build narrow row-location handoffs, including the optional two-pass path.

use super::*;

/// Discover stable positions using only local-predicate columns, then re-read
/// transfer keys at those positions and apply incoming membership. Wide query
/// payload remains deferred until formal execution.
pub(super) async fn collect_two_pass_row_location_handoff(
    table: &BloomTable,
    filters: &[CascadeFilter],
    required_columns: &[usize],
    input_row_hint: usize,
    log_transfer_steps: bool,
    parquet_layouts: &ParquetLayoutCache,
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
        try_prepare_location_plan(discovery_projection, log_transfer_steps, parquet_layouts)?
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

/// Form the narrow discovery pass from columns needed to prove P0's local
/// predicates; join keys and query payload are intentionally excluded here.
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

pub(super) fn row_location_transfer_plan(
    table: &BloomTable,
    filters: &[CascadeFilter],
    required_columns: &[usize],
    log_fallback: bool,
    parquet_layouts: &ParquetLayoutCache,
    context: &TaskContext,
) -> Result<Option<RowLocationTransferPlan>> {
    if required_columns.is_empty() {
        return Ok(None);
    }
    let projected = project_table_columns(table, required_columns, context)?;
    let Some(prepared) = try_prepare_location_plan(projected, log_fallback, parquet_layouts)?
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
