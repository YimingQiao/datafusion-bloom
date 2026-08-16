use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::common::test_util::batches_to_sort_string;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::SessionContext;
use datafusion_bloom::{BloomConfig, install_bloom};

fn context_with_bloom() -> Result<SessionContext> {
    let state = SessionStateBuilder::new_with_default_features().build();
    let state = install_bloom(
        state,
        BloomConfig {
            excitation_threshold: 1.01,
            ..BloomConfig::default()
        },
    )?;
    Ok(SessionContext::new_with_state(state))
}

fn count_named(plan: &Arc<dyn ExecutionPlan>, needle: &str) -> usize {
    usize::from(plan.name().contains(needle))
        + plan
            .children()
            .into_iter()
            .map(|child| count_named(child, needle))
            .sum::<usize>()
}

fn register_id_table(
    ctx: &SessionContext,
    name: &str,
    value_name: &str,
    ids: Vec<i64>,
    values: Vec<&str>,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(value_name, DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(values)),
        ],
    )?;
    ctx.register_table(
        name,
        Arc::new(MemTable::try_new(schema, vec![vec![batch]])?),
    )?;
    Ok(())
}

#[tokio::test]
async fn transfers_across_a_three_table_join_graph() -> Result<()> {
    let ctx = context_with_bloom()?;
    register_id_table(
        &ctx,
        "table_a",
        "av",
        vec![1, 2, 3, 4],
        vec!["a1", "a2", "a3", "a4"],
    )?;
    register_id_table(&ctx, "table_b", "bv", vec![2, 3, 5], vec!["b2", "b3", "b5"])?;
    register_id_table(&ctx, "table_c", "cv", vec![3, 5, 6], vec!["c3", "c5", "c6"])?;

    let dataframe = ctx
        .sql(
            "SELECT a.id, a.av, b.bv, c.cv \
             FROM table_a a \
             JOIN table_b b ON a.id = b.id \
             JOIN table_c c ON b.id = c.id",
        )
        .await?;
    let plan = dataframe.create_physical_plan().await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();

    assert_eq!(count_named(&plan, "HashJoinExec"), 2, "{formatted_plan}");
    assert_eq!(
        formatted_plan.matches("BloomCollection").count(),
        3,
        "{formatted_plan}"
    );

    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(
        batches_to_sort_string(&batches),
        [
            "+----+----+----+----+",
            "| id | av | bv | cv |",
            "+----+----+----+----+",
            "| 3  | a3 | b3 | c3 |",
            "+----+----+----+----+",
        ]
        .join("\n")
    );
    Ok(())
}

fn composite_schema(value_name: &str) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int64, false),
        Field::new("k2", DataType::Utf8, false),
        Field::new(value_name, DataType::Utf8, false),
    ]))
}

#[tokio::test]
async fn treats_multiple_join_keys_as_one_composite_key() -> Result<()> {
    let ctx = context_with_bloom()?;

    let left_schema = composite_schema("lv");
    let left = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 2])),
            Arc::new(StringArray::from(vec!["a", "b", "a", "b"])),
            Arc::new(StringArray::from(vec!["l1a", "l1b", "l2a", "l2b"])),
        ],
    )?;
    ctx.register_table(
        "composite_left",
        Arc::new(MemTable::try_new(left_schema, vec![vec![left]])?),
    )?;

    let right_schema = composite_schema("rv");
    let right = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 2])),
            Arc::new(StringArray::from(vec!["b", "a", "c"])),
            Arc::new(StringArray::from(vec!["r1b", "r2a", "r2c"])),
        ],
    )?;
    ctx.register_table(
        "composite_right",
        Arc::new(MemTable::try_new(right_schema, vec![vec![right]])?),
    )?;

    let dataframe = ctx
        .sql(
            "SELECT l.k1, l.k2, l.lv, r.rv \
             FROM composite_left l JOIN composite_right r \
             ON l.k1 = r.k1 AND l.k2 = r.k2",
        )
        .await?;
    let plan = dataframe.create_physical_plan().await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(count_named(&plan, "HashJoinExec"), 1, "{formatted_plan}");
    assert_eq!(
        formatted_plan.matches("BloomCollection").count(),
        2,
        "{formatted_plan}"
    );

    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(
        batches_to_sort_string(&batches),
        [
            "+----+----+-----+-----+",
            "| k1 | k2 | lv  | rv  |",
            "+----+----+-----+-----+",
            "| 1  | b  | l1b | r1b |",
            "| 2  | a  | l2a | r2a |",
            "+----+----+-----+-----+",
        ]
        .join("\n")
    );
    Ok(())
}
