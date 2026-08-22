use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::test_util::batches_to_sort_string;
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::physical_plan::parquet::ParquetAccessPlan;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::context::SessionConfig;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::properties::WriterProperties;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use datafusion_bloom::{BloomConfig, install_bloom};
use tempfile::TempDir;

fn schema(value_name: &str) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(value_name, DataType::Utf8, false),
    ]))
}

fn write_table(
    directory: &TempDir,
    filename: &str,
    value_name: &str,
    ids: Vec<i64>,
    values: Vec<&str>,
) -> Result<String> {
    let schema = schema(value_name);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(values)),
        ],
    )?;
    let path = directory.path().join(filename);
    let file = File::create(&path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(path.to_string_lossy().into_owned())
}

fn write_batch(path: &Path, batch: &RecordBatch) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

fn write_table_with_row_groups(
    directory: &TempDir,
    filename: &str,
    value_name: &str,
    ids: Vec<i64>,
    values: Vec<&str>,
    row_group_rows: usize,
) -> Result<String> {
    let schema = schema(value_name);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(values)),
        ],
    )?;
    let path = directory.path().join(filename);
    let file = File::create(&path)?;
    let properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(row_group_rows))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(properties))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(path.to_string_lossy().into_owned())
}

fn count_row_selected_sources(plan: &Arc<dyn ExecutionPlan>) -> usize {
    let current = plan
        .downcast_ref::<DataSourceExec>()
        .and_then(|source| source.data_source().downcast_ref::<FileScanConfig>())
        .is_some_and(|config| {
            config
                .file_groups
                .iter()
                .flat_map(|group| group.iter())
                .any(|file| file.extension::<ParquetAccessPlan>().is_some())
        });
    usize::from(current)
        + plan
            .children()
            .into_iter()
            .map(count_row_selected_sources)
            .sum::<usize>()
}

#[tokio::test]
async fn instant_sampling_preserves_query_local_parquet_semantics() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let left_path = write_table(
        &directory,
        "instant_left.parquet",
        "left_value",
        vec![1, 2, 3, 4],
        vec!["drop", "keep", "keep", "drop"],
    )?;
    let right_path = write_table(
        &directory,
        "instant_right.parquet",
        "right_value",
        vec![2, 3, 5],
        vec!["r2", "r3", "r5"],
    )?;

    let state = SessionStateBuilder::new_with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(1))
        .build();
    let state = install_bloom(
        state,
        BloomConfig {
            excitation_threshold: 1.01,
            ..BloomConfig::default()
        }
        .with_all_bounded_sources()
        .with_instant_sampling(),
    )?;
    let context = SessionContext::new_with_state(state);
    context
        .register_parquet("left_table", &left_path, ParquetReadOptions::default())
        .await?;
    context
        .register_parquet("right_table", &right_path, ParquetReadOptions::default())
        .await?;

    let batches = context
        .sql(
            "SELECT l.id, r.right_value \
             FROM left_table l JOIN right_table r ON l.id = r.id \
             WHERE l.left_value = 'keep'",
        )
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches_to_sort_string(&batches),
        [
            "+----+-------------+",
            "| id | right_value |",
            "+----+-------------+",
            "| 2  | r2          |",
            "| 3  | r3          |",
            "+----+-------------+",
        ]
        .join("\n")
    );
    Ok(())
}

#[tokio::test]
async fn instant_sampling_preserves_clustered_parquet_results() -> Result<()> {
    const ROW_GROUPS: usize = 8;
    const ROWS_PER_GROUP: usize = 10_000;
    const KEEP_PER_GROUP: usize = 25;
    let directory = tempfile::tempdir()?;
    let rows = ROW_GROUPS * ROWS_PER_GROUP;
    let ids = (0..rows as i64).collect::<Vec<_>>();
    let values = (0..rows)
        .map(|row| {
            if row % ROWS_PER_GROUP >= ROWS_PER_GROUP - KEEP_PER_GROUP {
                "keep"
            } else {
                "drop"
            }
        })
        .collect::<Vec<_>>();
    let left_path = write_table_with_row_groups(
        &directory,
        "clustered_left.parquet",
        "left_value",
        ids.clone(),
        values,
        ROWS_PER_GROUP,
    )?;
    let right_path = write_table_with_row_groups(
        &directory,
        "clustered_right.parquet",
        "right_value",
        ids,
        vec!["right"; rows],
        ROWS_PER_GROUP,
    )?;

    let state = SessionStateBuilder::new_with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .build();
    let state = install_bloom(
        state,
        BloomConfig {
            sample_rows: 64,
            excitation_threshold: 1.01,
            ..BloomConfig::default()
        }
        .with_all_bounded_sources()
        .with_instant_sampling()
        .with_instant_parquet_row_groups(ROW_GROUPS),
    )?;
    let context = SessionContext::new_with_state(state);
    context
        .register_parquet("left_table", &left_path, ParquetReadOptions::default())
        .await?;
    context
        .register_parquet("right_table", &right_path, ParquetReadOptions::default())
        .await?;

    let batches = context
        .sql(
            "SELECT count(*) AS matches \
             FROM left_table l JOIN right_table r ON l.id = r.id \
             WHERE l.left_value = 'keep'",
        )
        .await?
        .collect()
        .await?;
    assert_eq!(
        batches_to_sort_string(&batches),
        [
            "+---------+",
            "| matches |",
            "+---------+",
            "| 200     |",
            "+---------+",
        ]
        .join("\n")
    );
    Ok(())
}

#[tokio::test]
async fn instant_sampling_does_not_retain_a_session_sample() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let left_path = write_table(
        &directory,
        "sample_lifetime_left.parquet",
        "left_value",
        (0..20_000).collect(),
        (0..20_000)
            .map(|id| if id % 2 == 0 { "keep" } else { "drop" })
            .collect(),
    )?;
    let right_path = write_table(
        &directory,
        "sample_lifetime_right.parquet",
        "right_value",
        (0..20_000).collect(),
        vec!["right"; 20_000],
    )?;
    let query = "SELECT count(*) FROM left_table l JOIN right_table r ON l.id = r.id \
                 WHERE l.left_value = 'keep'";

    let make_context = |instant| -> Result<(SessionContext, Arc<GreedyMemoryPool>)> {
        let pool = Arc::new(GreedyMemoryPool::new(64 * 1024 * 1024));
        let runtime = Arc::new(
            RuntimeEnvBuilder::new()
                .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
                .build()?,
        );
        let state = SessionStateBuilder::new_with_default_features()
            .with_config(SessionConfig::new().with_target_partitions(1))
            .with_runtime_env(runtime)
            .build();
        let mut config = BloomConfig {
            excitation_threshold: 1.01,
            ..BloomConfig::default()
        }
        .with_all_bounded_sources();
        if instant {
            config = config.with_instant_sampling();
        }
        let state = install_bloom(state, config)?;
        Ok((SessionContext::new_with_state(state), pool))
    };

    let (prepared, prepared_pool) = make_context(false)?;
    let (instant, instant_pool) = make_context(true)?;
    for context in [&prepared, &instant] {
        context
            .register_parquet("left_table", &left_path, ParquetReadOptions::default())
            .await?;
        context
            .register_parquet("right_table", &right_path, ParquetReadOptions::default())
            .await?;
    }

    let prepared_plan = prepared.sql(query).await?.create_physical_plan().await?;
    drop(prepared_plan);
    assert!(
        prepared_pool.reserved() > 0,
        "prepared mode should retain its reusable source sample"
    );

    let instant_plan = instant.sql(query).await?.create_physical_plan().await?;
    drop(instant_plan);
    assert_eq!(
        instant_pool.reserved(),
        0,
        "instant mode should release its query-local sample with the plan"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_instant_queries_are_deterministic_and_release_memory() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let left_path = write_table(
        &directory,
        "concurrent_instant_left.parquet",
        "left_value",
        (0..20_000).collect(),
        (0..20_000)
            .map(|id| if id % 2 == 0 { "keep" } else { "drop" })
            .collect(),
    )?;
    let right_path = write_table(
        &directory,
        "concurrent_instant_right.parquet",
        "right_value",
        (0..20_000).collect(),
        vec!["right"; 20_000],
    )?;
    let pool = Arc::new(GreedyMemoryPool::new(256 * 1024 * 1024));
    let runtime = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::clone(&pool) as Arc<dyn MemoryPool>)
            .build()?,
    );
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_runtime_env(runtime)
        .build();
    let state = install_bloom(
        state,
        BloomConfig {
            excitation_threshold: 1.01,
            ..BloomConfig::default()
        }
        .with_all_bounded_sources()
        .with_instant_sampling(),
    )?;
    let context = SessionContext::new_with_state(state);
    context
        .register_parquet("left_table", &left_path, ParquetReadOptions::default())
        .await?;
    context
        .register_parquet("right_table", &right_path, ParquetReadOptions::default())
        .await?;

    let query = "SELECT count(*) AS matched \
                 FROM left_table l JOIN right_table r ON l.id = r.id \
                 WHERE l.left_value = 'keep'";
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..8 {
        let context = context.clone();
        tasks.spawn(async move {
            let plan = context.sql(query).await?.create_physical_plan().await?;
            let handoffs = displayable(plan.as_ref())
                .indent(false)
                .to_string()
                .matches("BloomCollection")
                .count();
            let batches = collect(plan, context.task_ctx()).await?;
            Ok::<_, DataFusionError>((handoffs, batches_to_sort_string(&batches)))
        });
    }

    while let Some(task) = tasks.join_next().await {
        let (handoffs, result) =
            task.map_err(|error| DataFusionError::ExecutionJoin(Box::new(error)))??;
        assert!(
            handoffs > 0,
            "Instant query should produce a FullRows handoff"
        );
        assert_eq!(
            result,
            [
                "+---------+",
                "| matched |",
                "+---------+",
                "| 10000   |",
                "+---------+",
            ]
            .join("\n")
        );
    }
    assert_eq!(
        pool.reserved(),
        0,
        "concurrent Instant queries should release samples and handoffs"
    );
    Ok(())
}

#[tokio::test]
async fn no_handoff_preserves_native_join_dynamic_filters() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let left_path = write_table(
        &directory,
        "left.parquet",
        "left_value",
        vec![1, 2, 3, 4],
        vec!["l1", "l2", "l3", "l4"],
    )?;
    let right_path = write_table(
        &directory,
        "right.parquet",
        "right_value",
        vec![2, 3, 5],
        vec!["r2", "r3", "r5"],
    )?;

    let mut session_config = SessionConfig::new();
    session_config
        .options_mut()
        .execution
        .parquet
        .pushdown_filters = true;

    let baseline_state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config.clone())
        .build();
    let baseline = SessionContext::new_with_state(baseline_state);
    let bloom_state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .build();
    let bloom_state = install_bloom(
        bloom_state,
        BloomConfig::default().with_all_bounded_sources(),
    )?;
    let bloom = SessionContext::new_with_state(bloom_state);

    for context in [&baseline, &bloom] {
        context
            .register_parquet("left_table", &left_path, ParquetReadOptions::default())
            .await?;
        context
            .register_parquet("right_table", &right_path, ParquetReadOptions::default())
            .await?;
    }

    let query = "SELECT l.id FROM left_table l \
                 WHERE EXISTS (SELECT 1 FROM right_table r WHERE r.id = l.id)";
    let baseline_plan = baseline.sql(query).await?.create_physical_plan().await?;
    let bloom_plan = bloom.sql(query).await?.create_physical_plan().await?;
    let baseline_text = displayable(baseline_plan.as_ref())
        .indent(false)
        .to_string();
    let bloom_text = displayable(bloom_plan.as_ref()).indent(false).to_string();

    let baseline_filters = baseline_text.matches("DynamicFilter").count();
    assert!(baseline_filters > 0, "{baseline_text}");
    assert_eq!(
        bloom_text.matches("DynamicFilter").count(),
        baseline_filters,
        "{bloom_text}"
    );
    assert!(!bloom_text.contains("BloomCollection"), "{bloom_text}");
    Ok(())
}

#[tokio::test]
async fn row_location_mode_keeps_narrow_materializations_as_full_rows() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let left_path = write_table(
        &directory,
        "left.parquet",
        "lv",
        vec![1, 2, 3, 4],
        vec!["l1", "l2", "l3", "l4"],
    )?;
    let right_path = write_table(
        &directory,
        "right.parquet",
        "rv",
        vec![3, 4, 5],
        vec!["r3", "r4", "r5"],
    )?;

    let state = SessionStateBuilder::new_with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(1))
        .build();
    let config = BloomConfig {
        excitation_threshold: 1.01,
        ..BloomConfig::default()
    }
    .with_all_bounded_sources()
    .with_row_locations();
    let state = install_bloom(state, config)?;
    let context = SessionContext::new_with_state(state);
    context
        .register_parquet("left_table", &left_path, ParquetReadOptions::default())
        .await?;
    context
        .register_parquet("right_table", &right_path, ParquetReadOptions::default())
        .await?;

    let dataframe = context
        .sql(
            "SELECT l.id, l.lv, r.rv \
             FROM left_table l JOIN right_table r ON l.id = r.id",
        )
        .await?;
    let plan = dataframe.create_physical_plan().await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(formatted_plan.matches("BloomCollection").count(), 2);
    assert_eq!(count_row_selected_sources(&plan), 0, "{formatted_plan}");

    let batches = collect(plan, context.task_ctx()).await?;
    assert_eq!(
        batches_to_sort_string(&batches),
        [
            "+----+----+----+",
            "| id | lv | rv |",
            "+----+----+----+",
            "| 3  | l3 | r3 |",
            "| 4  | l4 | r4 |",
            "+----+----+----+",
        ]
        .join("\n")
    );
    Ok(())
}

#[tokio::test]
async fn row_location_policy_rejects_small_wide_materializations() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let wide_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Utf8, false),
        Field::new("c", DataType::Utf8, false),
    ]));
    for (filename, ids, prefix) in [
        ("wide_left.parquet", vec![1, 2, 3, 4], "l"),
        ("wide_right.parquet", vec![3, 4, 5], "r"),
    ] {
        let values = |suffix| {
            (0..ids.len())
                .map(|index| format!("{prefix}{suffix}{index}"))
                .collect::<Vec<_>>()
        };
        let a = values("a");
        let b = values("b");
        let c = values("c");
        let batch = RecordBatch::try_new(
            Arc::clone(&wide_schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(a)),
                Arc::new(StringArray::from(b)),
                Arc::new(StringArray::from(c)),
            ],
        )?;
        let file = File::create(directory.path().join(filename))?;
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&wide_schema), None)?;
        writer.write(&batch)?;
        writer.close()?;
    }

    let state = SessionStateBuilder::new_with_default_features().build();
    let config = BloomConfig {
        excitation_threshold: 1.01,
        ..BloomConfig::default()
    }
    .with_all_bounded_sources()
    .with_row_locations();
    let state = install_bloom(state, config)?;
    let context = SessionContext::new_with_state(state);
    context
        .register_parquet(
            "wide_left",
            directory.path().join("wide_left.parquet").to_string_lossy(),
            ParquetReadOptions::default(),
        )
        .await?;
    context
        .register_parquet(
            "wide_right",
            directory
                .path()
                .join("wide_right.parquet")
                .to_string_lossy(),
            ParquetReadOptions::default(),
        )
        .await?;

    let plan = context
        .sql(
            "SELECT l.id, l.a, l.b, l.c, r.a, r.b, r.c \
             FROM wide_left l JOIN wide_right r ON l.id = r.id",
        )
        .await?
        .create_physical_plan()
        .await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    // Width alone is insufficient: at this scale the extra transfer scan and
    // formal Parquet re-read cannot amortize their fixed costs.
    assert_eq!(formatted_plan.matches("BloomCollection").count(), 2);
    assert_eq!(count_row_selected_sources(&plan), 0, "{formatted_plan}");

    let batches = collect(plan, context.task_ctx()).await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    Ok(())
}

#[tokio::test]
async fn cost_policy_uses_locations_for_a_large_wide_selective_handoff() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let seed_path = write_table(
        &directory,
        "seed.parquet",
        "flag",
        vec![7, 8],
        vec!["keep", "drop"],
    )?;
    let tail_path = write_table(
        &directory,
        "tail.parquet",
        "tail_value",
        vec![7, 9],
        vec!["match", "miss"],
    )?;

    // Reuse one high-entropy Arrow string array while writing eight physical
    // Parquet columns. The resulting sampled bytes/row make this genuinely
    // wide rather than merely wide in the logical schema.
    let row_count = 300_000;
    let mut fields = vec![
        Field::new("id", DataType::Int64, false),
        Field::new("local_flag", DataType::Utf8, false),
    ];
    fields.extend((0..8).map(|index| Field::new(format!("p{index}"), DataType::Utf8, false)));
    let wide_schema = Arc::new(Schema::new(fields));
    let ids = Arc::new(Int64Array::from_iter_values(0..row_count as i64)) as ArrayRef;
    let payload_values = (0..row_count)
        .map(|row| {
            let mixed = (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            format!("{mixed:016x}{:016x}", mixed.rotate_left(29))
        })
        .collect::<Vec<_>>();
    let payload = Arc::new(StringArray::from(payload_values)) as ArrayRef;
    let mut columns = vec![
        ids,
        Arc::new(StringArray::from(vec!["eligible"; row_count])) as ArrayRef,
    ];
    columns.extend((0..8).map(|_| Arc::clone(&payload)));
    let wide_batch = RecordBatch::try_new(Arc::clone(&wide_schema), columns)?;
    let wide_path = directory.path().join("wide.parquet");
    write_batch(&wide_path, &wide_batch)?;

    let state = SessionStateBuilder::new_with_default_features().build();
    let config = BloomConfig::default()
        .with_all_bounded_sources()
        .with_row_locations();
    let context = SessionContext::new_with_state(install_bloom(state, config)?);
    context
        .register_parquet("seed", &seed_path, ParquetReadOptions::default())
        .await?;
    context
        .register_parquet(
            "wide",
            wide_path.to_string_lossy(),
            ParquetReadOptions::default(),
        )
        .await?;
    context
        .register_parquet("tail", &tail_path, ParquetReadOptions::default())
        .await?;

    let plan = context
        .sql(
            "SELECT w.p0, w.p1, w.p2, w.p3, w.p4, w.p5, w.p6, w.p7, t.tail_value \
             FROM seed s \
             JOIN wide w ON s.id = w.id \
             JOIN tail t ON w.id = t.id \
             WHERE s.flag = 'keep' AND w.local_flag = 'eligible'",
        )
        .await?
        .create_physical_plan()
        .await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(count_row_selected_sources(&plan), 1, "{formatted_plan}");
    assert!(
        !formatted_plan.contains("local_flag"),
        "the selected formal scan re-read a verified local-filter column:\n{formatted_plan}"
    );

    let batches = collect(plan, context.task_ctx()).await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    Ok(())
}

#[tokio::test]
async fn default_path_materializes_all_query_columns() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let left_path = write_table(
        &directory,
        "left.parquet",
        "lv",
        vec![1, 2, 3, 4],
        vec!["l1", "l2", "l3", "l4"],
    )?;
    let right_path = write_table(
        &directory,
        "right.parquet",
        "rv",
        vec![3, 4, 5],
        vec!["r3", "r4", "r5"],
    )?;

    let state = SessionStateBuilder::new_with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(1))
        .build();
    let config = BloomConfig {
        excitation_threshold: 1.01,
        ..BloomConfig::default()
    }
    .with_all_bounded_sources();
    let state = install_bloom(state, config)?;
    let context = SessionContext::new_with_state(state);
    context
        .register_parquet("left_table", &left_path, ParquetReadOptions::default())
        .await?;
    context
        .register_parquet("right_table", &right_path, ParquetReadOptions::default())
        .await?;

    let plan = context
        .sql(
            "SELECT l.id, l.lv, r.rv \
             FROM left_table l JOIN right_table r ON l.id = r.id",
        )
        .await?
        .create_physical_plan()
        .await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(formatted_plan.matches("BloomCollection").count(), 2);
    assert_eq!(count_row_selected_sources(&plan), 0, "{formatted_plan}");

    let batches = collect(plan, context.task_ctx()).await?;
    assert_eq!(
        batches_to_sort_string(&batches),
        [
            "+----+----+----+",
            "| id | lv | rv |",
            "+----+----+----+",
            "| 3  | l3 | r3 |",
            "| 4  | l4 | r4 |",
            "+----+----+----+",
        ]
        .join("\n")
    );
    Ok(())
}

#[tokio::test]
async fn row_location_mode_preserves_single_aggregate_partitioning() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let part_directory = directory.path().join("part");
    let lineitem_directory = directory.path().join("lineitem");
    std::fs::create_dir(&part_directory)?;
    std::fs::create_dir(&lineitem_directory)?;

    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int64, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_container", DataType::Utf8, false),
    ]));
    for (file_index, keys, brands) in [
        (0, vec![1, 2], vec!["Target", "Other"]),
        (1, vec![3, 4], vec!["Target", "Other"]),
    ] {
        let batch = RecordBatch::try_new(
            Arc::clone(&part_schema),
            vec![
                Arc::new(Int64Array::from(keys)),
                Arc::new(StringArray::from(brands)),
                Arc::new(StringArray::from(vec!["Box", "Box"])),
            ],
        )?;
        write_batch(
            &part_directory.join(format!("part-{file_index}.parquet")),
            &batch,
        )?;
    }

    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int64, false),
        Field::new("l_quantity", DataType::Int64, false),
        Field::new("l_extendedprice", DataType::Int64, false),
    ]));
    for (file_index, keys, quantities, prices) in [
        (
            0,
            vec![1, 1, 2, 3],
            vec![1, 10, 5, 1],
            vec![100, 200, 300, 400],
        ),
        (
            1,
            vec![1, 3, 3, 4],
            vec![2, 10, 2, 1],
            vec![500, 600, 700, 800],
        ),
    ] {
        let batch = RecordBatch::try_new(
            Arc::clone(&lineitem_schema),
            vec![
                Arc::new(Int64Array::from(keys)),
                Arc::new(Int64Array::from(quantities)),
                Arc::new(Int64Array::from(prices)),
            ],
        )?;
        write_batch(
            &lineitem_directory.join(format!("lineitem-{file_index}.parquet")),
            &batch,
        )?;
    }

    let query = "\
        SELECT SUM(l.l_extendedprice) AS revenue \
        FROM part p \
        JOIN lineitem l ON p.p_partkey = l.l_partkey \
        JOIN ( \
            SELECT l_partkey, 0.5 * AVG(l_quantity) AS quantity_limit \
            FROM lineitem \
            GROUP BY l_partkey \
        ) a ON l.l_partkey = a.l_partkey \
        WHERE p.p_brand = 'Target' \
          AND p.p_container = 'Box' \
          AND l.l_quantity < a.quantity_limit";

    let baseline = SessionContext::new();
    baseline
        .register_parquet(
            "part",
            part_directory.to_string_lossy(),
            ParquetReadOptions::default(),
        )
        .await?;
    baseline
        .register_parquet(
            "lineitem",
            lineitem_directory.to_string_lossy(),
            ParquetReadOptions::default(),
        )
        .await?;
    let expected = baseline.sql(query).await?.collect().await?;

    let state = SessionStateBuilder::new_with_default_features().build();
    let config = BloomConfig {
        excitation_threshold: 1.01,
        ..BloomConfig::default()
    }
    .with_all_bounded_sources()
    .with_row_locations();
    let bloom = SessionContext::new_with_state(install_bloom(state, config)?);
    bloom
        .register_parquet(
            "part",
            part_directory.to_string_lossy(),
            ParquetReadOptions::default(),
        )
        .await?;
    bloom
        .register_parquet(
            "lineitem",
            lineitem_directory.to_string_lossy(),
            ParquetReadOptions::default(),
        )
        .await?;
    let plan = bloom.sql(query).await?.create_physical_plan().await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert_eq!(
        formatted_plan.matches("BloomCollection").count(),
        2,
        "{formatted_plan}"
    );
    let actual = collect(plan, bloom.task_ctx()).await?;

    assert_eq!(
        batches_to_sort_string(&actual),
        batches_to_sort_string(&expected)
    );
    Ok(())
}

#[tokio::test]
async fn direct_membership_uses_the_parquet_reader_by_default() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let a_directory = directory.path().join("a");
    let b_directory = directory.path().join("b");
    let c_directory = directory.path().join("c");
    std::fs::create_dir(&a_directory)?;
    std::fs::create_dir(&b_directory)?;
    std::fs::create_dir(&c_directory)?;

    let a_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int64, false),
        Field::new("flag", DataType::Utf8, false),
    ]));
    write_batch(
        &a_directory.join("a.parquet"),
        &RecordBatch::try_new(
            Arc::clone(&a_schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["keep", "drop"])),
            ],
        )?,
    )?;

    let b_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int64, false),
        Field::new("c_id", DataType::Int64, false),
    ]));
    write_batch(
        &b_directory.join("b.parquet"),
        &RecordBatch::try_new(
            Arc::clone(&b_schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(Int64Array::from(vec![42, 43])),
            ],
        )?,
    )?;

    let c_schema = Arc::new(Schema::new(vec![
        Field::new("c_id", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    write_batch(
        &c_directory.join("c.parquet"),
        &RecordBatch::try_new(
            Arc::clone(&c_schema),
            vec![
                Arc::new(Int64Array::from(vec![42; 100])),
                Arc::new(StringArray::from(vec!["payload"; 100])),
            ],
        )?,
    )?;

    let state = SessionStateBuilder::new_with_default_features().build();
    let config = BloomConfig {
        excitation_threshold: 1.0,
        ..BloomConfig::default()
    }
    .with_all_bounded_sources()
    .with_row_locations();
    let context = SessionContext::new_with_state(install_bloom(state, config)?);
    for (name, path) in [
        ("table_a", &a_directory),
        ("table_b", &b_directory),
        ("table_c", &c_directory),
    ] {
        context
            .register_parquet(name, path.to_string_lossy(), ParquetReadOptions::default())
            .await?;
    }

    let plan = context
        .sql(
            "SELECT COUNT(*) AS matches \
             FROM table_a a \
             JOIN table_b b ON a.a_id = b.a_id \
             JOIN table_c c ON b.c_id = c.c_id \
             WHERE a.flag = 'keep'",
        )
        .await?
        .create_physical_plan()
        .await?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        !formatted_plan.contains("BloomScanBoundaryExec"),
        "{formatted_plan}"
    );
    assert!(
        formatted_plan.contains("BloomTransferMembership"),
        "{formatted_plan}"
    );
    assert!(
        formatted_plan
            .lines()
            .filter(|line| line.contains("DataSourceExec"))
            .any(|line| line.contains("BloomTransferMembership")),
        "{formatted_plan}"
    );
    assert!(
        formatted_plan
            .lines()
            .filter(|line| line.contains("DataSourceExec"))
            .any(|line| line.contains("c_id@0 = 42")),
        "{formatted_plan}"
    );

    let batches = collect(Arc::clone(&plan), context.task_ctx()).await?;
    assert_eq!(
        batches_to_sort_string(&batches),
        [
            "+---------+",
            "| matches |",
            "+---------+",
            "| 100     |",
            "+---------+",
        ]
        .join("\n")
    );
    Ok(())
}
