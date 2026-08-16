use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::common::test_util::batches_to_sort_string;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::SessionContext;
use datafusion_bloom::{BloomConfig, BloomQueryPlanner, install_bloom};

fn context_with_bloom() -> Result<SessionContext> {
    context_with_config(BloomConfig {
        excitation_threshold: 1.01,
        ..BloomConfig::default()
    })
}

fn context_with_config(config: BloomConfig) -> Result<SessionContext> {
    let state = SessionStateBuilder::new_with_default_features().build();
    let state = install_bloom(state, config)?;
    Ok(SessionContext::new_with_state(state))
}

fn register_tables(ctx: &SessionContext) -> Result<()> {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("left_value", DataType::Utf8, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(2),
                Some(2),
                Some(3),
                Some(5),
                None,
            ])),
            Arc::new(StringArray::from(vec![
                "l1", "l2a", "l2b", "l3", "l5", "ln",
            ])),
        ],
    )?;
    ctx.register_table(
        "left_table",
        Arc::new(MemTable::try_new(left_schema, vec![vec![left_batch]])?),
    )?;

    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("right_value", DataType::Utf8, false),
    ]));
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int64Array::from(vec![
                Some(2),
                Some(2),
                Some(4),
                Some(5),
                None,
            ])),
            Arc::new(StringArray::from(vec!["r2a", "r2b", "r4", "r5", "rn"])),
        ],
    )?;
    ctx.register_table(
        "right_table",
        Arc::new(MemTable::try_new(right_schema, vec![vec![right_batch]])?),
    )?;
    Ok(())
}

fn count_named(plan: &Arc<dyn ExecutionPlan>, needle: &str) -> usize {
    usize::from(plan.name().contains(needle))
        + plan
            .children()
            .into_iter()
            .map(|child| count_named(child, needle))
            .sum::<usize>()
}

#[tokio::test]
async fn materializes_both_inputs_and_keeps_stock_hash_join() -> Result<()> {
    let ctx = context_with_bloom()?;
    register_tables(&ctx)?;

    let dataframe = ctx
        .sql(
            "SELECT l.id, l.left_value, r.right_value \
             FROM left_table l JOIN right_table r ON l.id = r.id",
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
            "+----+------------+-------------+",
            "| id | left_value | right_value |",
            "+----+------------+-------------+",
            "| 2  | l2a        | r2a         |",
            "| 2  | l2a        | r2b         |",
            "| 2  | l2b        | r2a         |",
            "| 2  | l2b        | r2b         |",
            "| 5  | l5         | r5          |",
            "+----+------------+-------------+",
        ]
        .join("\n")
    );
    Ok(())
}

#[tokio::test]
async fn empty_join_input_is_left_safe() -> Result<()> {
    let ctx = context_with_bloom()?;

    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let non_empty = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
    )?;
    ctx.register_table(
        "non_empty",
        Arc::new(MemTable::try_new(
            Arc::clone(&schema),
            vec![vec![non_empty]],
        )?),
    )?;
    ctx.register_table(
        "empty_table",
        Arc::new(MemTable::try_new(schema, vec![vec![]])?),
    )?;

    let dataframe = ctx
        .sql("SELECT * FROM non_empty n JOIN empty_table e ON n.id = e.id")
        .await?;
    let plan = dataframe.create_physical_plan().await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    // A physically empty source has no useful outgoing transfer work; the
    // adaptive scheduler leaves DataFusion's already-correct empty join alone.
    assert_eq!(formatted_plan.matches("BloomCollection").count(), 0);

    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    Ok(())
}

#[tokio::test]
async fn leaves_unsupported_outer_join_untouched() -> Result<()> {
    let ctx = context_with_bloom()?;
    register_tables(&ctx)?;

    let dataframe = ctx
        .sql(
            "SELECT l.id, l.left_value, r.right_value \
             FROM left_table l LEFT JOIN right_table r ON l.id = r.id",
        )
        .await?;
    let plan = dataframe.create_physical_plan().await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(formatted_plan.matches("BloomCollection").count(), 0);

    let batches = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 8);
    Ok(())
}

#[tokio::test]
async fn disabled_config_delegates_to_datafusion() -> Result<()> {
    let ctx = context_with_config(BloomConfig {
        enabled: false,
        ..BloomConfig::default()
    })?;
    register_tables(&ctx)?;

    let plan = ctx
        .sql("SELECT * FROM left_table l JOIN right_table r ON l.id = r.id")
        .await?
        .create_physical_plan()
        .await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(formatted_plan.matches("BloomCollection").count(), 0);
    assert_eq!(count_named(&plan, "HashJoinExec"), 1);
    Ok(())
}

#[test]
fn invalid_config_is_rejected() {
    let error = BloomQueryPlanner::new(BloomConfig {
        false_positive_rate: 1.0,
        ..BloomConfig::default()
    })
    .expect_err("an invalid false-positive rate must be rejected");
    assert!(error.to_string().contains("false_positive_rate"));
}
