use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::common::test_util::batches_to_sort_string;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
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

fn context_with_memory_limit(bytes: usize) -> Result<(SessionContext, Arc<GreedyMemoryPool>)> {
    let pool = Arc::new(GreedyMemoryPool::new(bytes));
    let runtime = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
            .build()?,
    );
    let state = SessionStateBuilder::new_with_default_features()
        .with_runtime_env(runtime)
        .build();
    let state = install_bloom(
        state,
        BloomConfig {
            excitation_threshold: 1.01,
            ..BloomConfig::default()
        },
    )?;
    Ok((SessionContext::new_with_state(state), pool))
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

fn register_wide_tables(ctx: &SessionContext) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let ids = (0..256_i64).collect::<Vec<_>>();
    let payloads = (0..256)
        .map(|index| format!("{index:04}-{}", "x".repeat(252)))
        .collect::<Vec<_>>();
    for name in ["wide_left", "wide_right"] {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ids.clone())),
                Arc::new(StringArray::from(payloads.clone())),
            ],
        )?;
        ctx.register_table(
            name,
            Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![batch]])?),
        )?;
    }
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

#[tokio::test]
async fn insufficient_handoff_memory_falls_back_to_native_plan() -> Result<()> {
    let (ctx, pool) = context_with_memory_limit(1)?;
    register_tables(&ctx)?;

    let plan = ctx
        .sql(
            "SELECT l.id, l.left_value, r.right_value \
             FROM left_table l JOIN right_table r ON l.id = r.id",
        )
        .await?
        .create_physical_plan()
        .await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(
        formatted_plan.matches("BloomCollection").count(),
        0,
        "{formatted_plan}"
    );
    assert_eq!(count_named(&plan, "HashJoinExec"), 1, "{formatted_plan}");
    assert_eq!(pool.reserved(), 0, "failed handoffs must release memory");
    Ok(())
}

#[tokio::test]
async fn native_execution_can_succeed_after_handoff_memory_fallback() -> Result<()> {
    let query = "SELECT l.id, l.payload, r.payload \
                 FROM wide_left l JOIN wide_right r ON l.id = r.id";
    let mut successful_limit = None;
    for limit in [80, 96, 112, 128, 160, 192, 256]
        .into_iter()
        .map(|kib| kib * 1024)
    {
        let (ctx, _) = context_with_memory_limit(limit)?;
        register_wide_tables(&ctx)?;
        let plan = ctx.sql(query).await?.create_physical_plan().await?;
        let is_native = count_named(&plan, "BloomCollection") == 0;
        if !is_native {
            continue;
        }
        if let Ok(batches) = collect(plan, ctx.task_ctx()).await
            && batches.iter().map(RecordBatch::num_rows).sum::<usize>() == 256
        {
            successful_limit = Some(limit);
            break;
        }
    }
    assert!(
        successful_limit.is_some(),
        "no tested memory limit allowed native execution after Bloom fallback"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_queries_remain_deterministic() -> Result<()> {
    let ctx = Arc::new(context_with_bloom()?);
    register_tables(&ctx)?;
    let query = "SELECT l.id, l.left_value, r.right_value \
                 FROM left_table l JOIN right_table r ON l.id = r.id";
    let expected = batches_to_sort_string(&ctx.sql(query).await?.collect().await?);

    let jobs = (0..16).map(|_| {
        let ctx = Arc::clone(&ctx);
        let expected = expected.clone();
        tokio::spawn(async move {
            let actual = ctx.sql(query).await?.collect().await?;
            assert_eq!(batches_to_sort_string(&actual), expected);
            Ok::<_, datafusion::common::DataFusionError>(())
        })
    });
    for result in futures::future::join_all(jobs).await {
        result.expect("query task")?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repeated_queries_release_every_handoff_reservation() -> Result<()> {
    let (ctx, pool) = context_with_memory_limit(8 * 1024 * 1024)?;
    register_tables(&ctx)?;
    let query = "SELECT l.id, l.left_value, r.right_value \
                 FROM left_table l JOIN right_table r ON l.id = r.id";

    let plan = ctx.sql(query).await?.create_physical_plan().await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(
        formatted_plan.matches("BloomCollection").count(),
        2,
        "{formatted_plan}"
    );
    drop(plan);
    assert_eq!(pool.reserved(), 0);

    for _ in 0..100 {
        let batches = ctx.sql(query).await?.collect().await?;
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 5);
        assert_eq!(pool.reserved(), 0, "query leaked a memory reservation");
    }
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
