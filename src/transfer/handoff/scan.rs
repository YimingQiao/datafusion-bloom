//! Place transfer membership into table scans without disturbing P0 predicates.

use super::*;

/// Remove local predicates only after retained locations prove those rows have
/// already passed P0. Projections are reconstructed so the table-operator
/// schema remains unchanged for the subsequent selected re-read.
pub(in crate::transfer) fn strip_verified_local_filters(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
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

pub(super) fn strip_parquet_source_predicates(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
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

/// Keep independently built transfer predicates as sequential stages so cheap
/// membership can reduce rows before expensive local/string expressions.
pub(super) fn transfer_filter_plan(
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

/// Collect all keys a row-location handoff may need in later propagation
/// rounds. Omitting one would make the physical representation non-transitive.
pub(in crate::transfer) fn required_join_columns(
    graph: &BloomGraph,
    table: TableId,
) -> Result<Vec<usize>> {
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

/// Measure widths only when cost-based late materialization is enabled;
/// FullRows does not require sampling to establish its semantics.
pub(in crate::transfer) fn materialization_widths(
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

/// Classify ordering cost conservatively: transferred membership is cheap,
/// while expressions touching variable-width payload should run last.
pub(in crate::transfer) fn predicate_is_expensive(
    predicate: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> bool {
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

pub(super) fn transfer_scan_plan(
    plan: Arc<dyn ExecutionPlan>,
    filters: &[CascadeFilter],
    context: &TaskContext,
    parquet_membership_placement: ParquetMembershipPlacement,
) -> Result<Arc<dyn ExecutionPlan>> {
    transfer_scan_plan_impl(plan, filters, context, false, parquet_membership_placement)
}

pub(in crate::transfer) fn formal_transfer_scan_plan(
    plan: Arc<dyn ExecutionPlan>,
    filters: &[CascadeFilter],
    context: &TaskContext,
    parquet_membership_placement: ParquetMembershipPlacement,
) -> Result<Arc<dyn ExecutionPlan>> {
    transfer_scan_plan_impl(plan, filters, context, true, parquet_membership_placement)
}

/// Add only Bloom's incremental predicates to an already optimized P0 tree.
/// Existing local predicates retain their meaning and placement; the formal
/// boundary controls whether membership may enter the Parquet reader.
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

/// Derive redundant Parquet pruning bounds from exact integer membership.
/// These bounds may discard storage regions, but membership remains the
/// semantic predicate that decides which rows survive transfer.
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
