use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::common::hash_utils::{RandomState, create_hashes};
use datafusion::datasource::physical_plan::FileScanConfig;
use datafusion::datasource::physical_plan::parquet::ParquetAccessPlan;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::SessionStateBuilder;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_bloom::{BloomConfig, install_bloom};
use tempfile::TempDir;

const ROWS: usize = 300_000;
const RUNS: usize = 5;
const SQL: &str = "\
    SELECT w.p0, w.p1, w.p2, w.p3, w.p4, w.p5, w.p6, w.p7, t.tail_value \
    FROM seed s \
    JOIN wide w ON s.id = w.id \
    JOIN tail t ON w.id = t.id \
    WHERE s.flag = 'keep'";

#[derive(Debug)]
struct Measurement {
    elapsed: Duration,
    rows: usize,
    hash_sum: u64,
    hash_xor: u64,
    full_rows: usize,
    row_locations: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let data = tempfile::tempdir()?;
    prepare_data(&data)?;

    let baseline = make_context(data.path(), None).await?;
    let full_rows = make_context(
        data.path(),
        Some(BloomConfig::default().with_all_bounded_sources()),
    )
    .await?;
    let row_locations = make_context(
        data.path(),
        Some(
            BloomConfig::default()
                .with_all_bounded_sources()
                .with_row_locations(),
        ),
    )
    .await?;

    // Warm prepared samples, metadata, and OS page cache before comparing the
    // two handoff representations.
    let expected = measure(&baseline).await?;
    assert_measurement(&expected, 1, 0, 0);
    let full_warmup = measure(&full_rows).await?;
    assert_measurement(&full_warmup, 1, 3, 0);
    let row_warmup = measure(&row_locations).await?;
    assert_measurement(&row_warmup, 1, 2, 1);

    let mut baseline_runs = Vec::with_capacity(RUNS);
    let mut full_runs = Vec::with_capacity(RUNS);
    let mut row_runs = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        match run % 3 {
            0 => {
                baseline_runs.push(measure(&baseline).await?);
                full_runs.push(measure(&full_rows).await?);
                row_runs.push(measure(&row_locations).await?);
            }
            1 => {
                row_runs.push(measure(&row_locations).await?);
                baseline_runs.push(measure(&baseline).await?);
                full_runs.push(measure(&full_rows).await?);
            }
            _ => {
                full_runs.push(measure(&full_rows).await?);
                row_runs.push(measure(&row_locations).await?);
                baseline_runs.push(measure(&baseline).await?);
            }
        }
    }

    for measurement in baseline_runs.iter().chain(&full_runs).chain(&row_runs) {
        assert_eq!(measurement.rows, expected.rows);
        assert_eq!(measurement.hash_sum, expected.hash_sum);
        assert_eq!(measurement.hash_xor, expected.hash_xor);
    }
    let baseline_ms = median_ms(&baseline_runs);
    let full_ms = median_ms(&full_runs);
    let row_ms = median_ms(&row_runs);
    println!("wide selective Parquet handoff benchmark");
    println!("rows={ROWS} payload_columns=8 selected_rows=1 runs={RUNS}");
    println!("target_partitions=64 join_dynamic_filters=false batch_size=8192");
    println!("mode\tmedian_ms\tvs_datafusion\tvs_full_rows\tfull_rows\trow_locations");
    println!(
        "DataFusion\t{baseline_ms:.3}\t1.000x\t{:.3}x\t0\t0",
        full_ms / baseline_ms
    );
    println!(
        "Bloom FullRows\t{full_ms:.3}\t{:.3}x\t1.000x\t3\t0",
        baseline_ms / full_ms
    );
    println!(
        "Bloom RowLocations\t{row_ms:.3}\t{:.3}x\t{:.3}x\t2\t1",
        baseline_ms / row_ms,
        full_ms / row_ms
    );
    Ok(())
}

async fn make_context(path: &Path, bloom: Option<BloomConfig>) -> Result<SessionContext> {
    // This benchmark isolates handoff representation. Disable DataFusion's
    // native dynamic predicate in every mode so FullRows and RowLocations
    // execute the same transfer graph.
    let mut state_config = SessionConfig::new().with_target_partitions(64);
    state_config
        .options_mut()
        .optimizer
        .enable_join_dynamic_filter_pushdown = false;
    let mut state = SessionStateBuilder::new_with_default_features()
        .with_config(state_config)
        .build();
    if let Some(config) = bloom {
        state = install_bloom(state, config)?;
    }
    let context = SessionContext::new_with_state(state);
    for table in ["seed", "wide", "tail"] {
        context
            .register_parquet(
                table,
                path.join(format!("{table}.parquet")).to_string_lossy(),
                ParquetReadOptions::default(),
            )
            .await?;
    }
    Ok(context)
}

async fn measure(context: &SessionContext) -> Result<Measurement> {
    let started = Instant::now();
    let plan = context.sql(SQL).await?.create_physical_plan().await?;
    let batches = collect(Arc::clone(&plan), context.task_ctx()).await?;
    let elapsed = started.elapsed();
    let formatted = displayable(plan.as_ref()).indent(false).to_string();
    let (rows, hash_sum, hash_xor) = fingerprint(&batches)?;
    Ok(Measurement {
        elapsed,
        rows,
        hash_sum,
        hash_xor,
        full_rows: formatted.matches("BloomCollection").count(),
        row_locations: count_row_selected_sources(&plan),
    })
}

fn fingerprint(batches: &[RecordBatch]) -> Result<(usize, u64, u64)> {
    let random_state = RandomState::with_seed(0x424c_4f4f_4d44_4631);
    let mut rows = 0;
    let mut hash_sum = 0_u64;
    let mut hash_xor = 0_u64;
    for batch in batches {
        let mut hashes = vec![0; batch.num_rows()];
        create_hashes(batch.columns(), &random_state, &mut hashes)?;
        rows += hashes.len();
        for hash in hashes {
            hash_sum = hash_sum.wrapping_add(hash);
            hash_xor ^= hash;
        }
    }
    Ok((rows, hash_sum, hash_xor))
}

fn assert_measurement(
    measurement: &Measurement,
    rows: usize,
    full_rows: usize,
    row_locations: usize,
) {
    assert_eq!(measurement.rows, rows);
    assert_eq!(measurement.full_rows, full_rows);
    assert_eq!(measurement.row_locations, row_locations);
}

fn median_ms(measurements: &[Measurement]) -> f64 {
    let mut values = measurements
        .iter()
        .map(|measurement| measurement.elapsed.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
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

fn prepare_data(directory: &TempDir) -> Result<()> {
    write_small_table(
        &directory.path().join("seed.parquet"),
        "flag",
        vec![7, 8],
        vec!["keep", "drop"],
    )?;
    write_small_table(
        &directory.path().join("tail.parquet"),
        "tail_value",
        vec![7, 9],
        vec!["match", "miss"],
    )?;

    let mut fields = vec![Field::new("id", DataType::Int64, false)];
    fields.extend((0..8).map(|index| Field::new(format!("p{index}"), DataType::Utf8, false)));
    let schema = Arc::new(Schema::new(fields));
    let ids = Arc::new(Int64Array::from_iter_values(0..ROWS as i64)) as ArrayRef;
    let payload_values = (0..ROWS)
        .map(|row| {
            let mixed = (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            format!("{mixed:016x}{:016x}", mixed.rotate_left(29))
        })
        .collect::<Vec<_>>();
    let payload = Arc::new(StringArray::from(payload_values)) as ArrayRef;
    let mut columns = vec![ids];
    columns.extend((0..8).map(|_| Arc::clone(&payload)));
    write_batch(
        &directory.path().join("wide.parquet"),
        &RecordBatch::try_new(schema, columns)?,
    )
}

fn write_small_table(
    path: &Path,
    value_name: &str,
    ids: Vec<i64>,
    values: Vec<&str>,
) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(value_name, DataType::Utf8, false),
    ]));
    write_batch(
        path,
        &RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(values)),
            ],
        )?,
    )
}

fn write_batch(path: &Path, batch: &RecordBatch) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}
