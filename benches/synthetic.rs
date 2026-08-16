use std::cmp::Ordering;
use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::array::{ArrayRef, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{Result, ScalarValue};
use datafusion::datasource::MemTable;
use datafusion::datasource::TableProvider;
use datafusion::execution::{SessionStateBuilder, TaskContext};
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_bloom::{BloomConfig, install_bloom};

#[derive(Debug, Clone)]
struct Options {
    topology: String,
    fact_rows: usize,
    dimension_rows: usize,
    dimensions: usize,
    selectivity_bps: u64,
    partitions: usize,
    batch_size: usize,
    warmups: usize,
    runs: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            topology: "star".to_string(),
            fact_rows: 4_000_000,
            dimension_rows: 100_000,
            dimensions: 4,
            selectivity_bps: 500,
            partitions: 8,
            batch_size: 16_384,
            warmups: 2,
            runs: 7,
        }
    }
}

impl Options {
    fn parse() -> Self {
        let mut options = Self::default();
        let mut arguments = env::args().skip(1);
        while let Some(flag) = arguments.next() {
            // Cargo passes this marker to custom benchmark binaries.
            if flag == "--bench" {
                continue;
            }
            let value = arguments
                .next()
                .unwrap_or_else(|| panic!("missing value after {flag}"));
            match flag.as_str() {
                "--topology" => options.topology = value,
                "--fact-rows" => options.fact_rows = parse(&flag, &value),
                "--dimension-rows" => options.dimension_rows = parse(&flag, &value),
                "--dimensions" => options.dimensions = parse(&flag, &value),
                "--selectivity-bps" => options.selectivity_bps = parse(&flag, &value),
                "--partitions" => options.partitions = parse(&flag, &value),
                "--batch-size" => options.batch_size = parse(&flag, &value),
                "--warmups" => options.warmups = parse(&flag, &value),
                "--runs" => options.runs = parse(&flag, &value),
                _ => panic!("unknown argument: {flag}"),
            }
        }
        assert!(options.fact_rows > 0);
        assert!(matches!(options.topology.as_str(), "star" | "balanced"));
        assert!(options.dimension_rows > 0);
        assert!((2..=8).contains(&options.dimensions));
        assert!((1..=10_000).contains(&options.selectivity_bps));
        assert!(options.partitions > 0);
        assert!(options.batch_size > 0);
        assert!(options.runs > 0);
        options
    }
}

#[derive(Debug)]
struct Measurement {
    planning: Duration,
    execution: Duration,
    result: ScalarValue,
}

impl Measurement {
    fn total(&self) -> Duration {
        self.planning + self.execution
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let options = Options::parse();
    let tables = match options.topology.as_str() {
        "star" => build_star(&options)?,
        "balanced" => build_balanced(&options)?,
        _ => unreachable!(),
    };
    let baseline = make_context(&options, &tables, false)?;
    let bloom = make_context(&options, &tables, true)?;
    let sql = match options.topology.as_str() {
        "star" => star_query(&options),
        "balanced" => balanced_query(&options),
        _ => unreachable!(),
    };

    let baseline_plan = create_plan(&baseline, &sql).await?;
    let bloom_plan = create_plan(&bloom, &sql).await?;
    let baseline_hash_joins = count_named(&baseline_plan, "HashJoinExec");
    let bloom_hash_joins = count_named(&bloom_plan, "HashJoinExec");
    let formatted_bloom = displayable(bloom_plan.as_ref()).indent(false).to_string();
    let collections = formatted_bloom.matches("BloomCollection").count();
    if env::var_os("BLOOM_BENCH_SHOW_PLAN").is_some() {
        println!(
            "DataFusion plan:\n{}",
            displayable(baseline_plan.as_ref()).indent(false)
        );
        println!("Bloom plan:\n{formatted_bloom}");
    }

    for _ in 0..options.warmups {
        let left = measure(&baseline, &sql).await?;
        let right = measure(&bloom, &sql).await?;
        assert_eq!(left.result, right.result);
    }

    let mut baseline_runs = Vec::with_capacity(options.runs);
    let mut bloom_runs = Vec::with_capacity(options.runs);
    for run in 0..options.runs {
        if run % 2 == 0 {
            baseline_runs.push(measure(&baseline, &sql).await?);
            bloom_runs.push(measure(&bloom, &sql).await?);
        } else {
            bloom_runs.push(measure(&bloom, &sql).await?);
            baseline_runs.push(measure(&baseline, &sql).await?);
        }
        assert_eq!(baseline_runs[run].result, bloom_runs[run].result);
    }

    let baseline_plan_ms = median_ms(&baseline_runs, |run| run.planning);
    let baseline_exec_ms = median_ms(&baseline_runs, |run| run.execution);
    let baseline_total_ms = median_ms(&baseline_runs, Measurement::total);
    let bloom_plan_ms = median_ms(&bloom_runs, |run| run.planning);
    let bloom_exec_ms = median_ms(&bloom_runs, |run| run.execution);
    let bloom_total_ms = median_ms(&bloom_runs, Measurement::total);

    println!("Bloom synthetic {} benchmark", options.topology);
    println!(
        "topology={} fact_rows={} dimension_rows={} dimensions={} selectivity={:.2}% partitions={} batch_size={} warmups={} runs={}",
        options.topology,
        options.fact_rows,
        options.dimension_rows,
        options.dimensions,
        options.selectivity_bps as f64 / 100.0,
        options.partitions,
        options.batch_size,
        options.warmups,
        options.runs
    );
    println!(
        "result={} baseline_hash_joins={baseline_hash_joins} bloom_hash_joins={bloom_hash_joins} bloom_collections={collections}",
        baseline_runs[0].result
    );
    println!();
    println!("| Mode | Planning/transfer median | Execution median | End-to-end median |");
    println!("|---|---:|---:|---:|");
    println!(
        "| DataFusion | {baseline_plan_ms:.3} ms | {baseline_exec_ms:.3} ms | {baseline_total_ms:.3} ms |"
    );
    println!("| Bloom | {bloom_plan_ms:.3} ms | {bloom_exec_ms:.3} ms | {bloom_total_ms:.3} ms |");
    println!();
    println!(
        "end_to_end_speedup={:.3}x execution_speedup={:.3}x",
        baseline_total_ms / bloom_total_ms,
        baseline_exec_ms / bloom_exec_ms
    );

    Ok(())
}

async fn create_plan(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

async fn measure(ctx: &SessionContext, sql: &str) -> Result<Measurement> {
    let start = Instant::now();
    let plan = create_plan(ctx, sql).await?;
    let planned = Instant::now();
    let batches = collect(plan, task_context(ctx)).await?;
    let finished = Instant::now();
    let result = ScalarValue::try_from_array(batches[0].column(0), 0)?;
    Ok(Measurement {
        planning: planned.duration_since(start),
        execution: finished.duration_since(planned),
        result,
    })
}

fn task_context(ctx: &SessionContext) -> Arc<TaskContext> {
    ctx.task_ctx()
}

fn make_context(
    options: &Options,
    tables: &[(String, Arc<dyn TableProvider>)],
    bloom: bool,
) -> Result<SessionContext> {
    let session_config = SessionConfig::new()
        .with_target_partitions(options.partitions)
        .with_batch_size(options.batch_size);
    let mut state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .build();
    if bloom {
        state = install_bloom(state, BloomConfig::default())?;
    }
    let context = SessionContext::new_with_state(state);
    for (name, table) in tables {
        context.register_table(name, Arc::clone(table))?;
    }
    Ok(context)
}

fn build_star(options: &Options) -> Result<Vec<(String, Arc<dyn TableProvider>)>> {
    let mut tables: Vec<(String, Arc<dyn TableProvider>)> =
        Vec::with_capacity(options.dimensions + 1);
    let fact_fields = (0..options.dimensions)
        .map(|index| Field::new(format!("k{index}"), DataType::UInt64, false))
        .chain(std::iter::once(Field::new(
            "payload",
            DataType::UInt64,
            false,
        )))
        .collect::<Vec<_>>();
    let fact_schema = Arc::new(Schema::new(fact_fields));
    let mut fact_partitions = vec![vec![]; options.partitions];
    let mut start = 0;
    while start < options.fact_rows {
        let end = (start + options.batch_size).min(options.fact_rows);
        let mut columns: Vec<Vec<u64>> = (0..options.dimensions + 1)
            .map(|_| Vec::with_capacity(end - start))
            .collect();
        for row in start..end {
            for (dimension, column) in columns[..options.dimensions].iter_mut().enumerate() {
                column.push(
                    mix64(row as u64 ^ ((dimension as u64 + 1) * 0x9e37_79b9))
                        % options.dimension_rows as u64,
                );
            }
            columns[options.dimensions].push(mix64(row as u64));
        }
        let arrays = columns
            .into_iter()
            .map(|values| Arc::new(UInt64Array::from(values)) as ArrayRef)
            .collect();
        let batch = RecordBatch::try_new(Arc::clone(&fact_schema), arrays)?;
        let partition = (start / options.batch_size) % options.partitions;
        fact_partitions[partition].push(batch);
        start = end;
    }
    tables.push((
        "fact".to_string(),
        Arc::new(MemTable::try_new(fact_schema, fact_partitions)?),
    ));

    let dimension_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("marker", DataType::UInt64, false),
        Field::new("payload", DataType::UInt64, false),
    ]));
    for dimension in 0..options.dimensions {
        let mut partitions = vec![vec![]; options.partitions];
        let mut start = 0;
        while start < options.dimension_rows {
            let end = (start + options.batch_size).min(options.dimension_rows);
            let ids = (start as u64..end as u64).collect::<Vec<_>>();
            let markers = (start as u64..end as u64)
                .map(|id| mix64(id ^ ((dimension as u64 + 11) * 0x85eb_ca6b)) % 10_000)
                .collect::<Vec<_>>();
            let payload = (start as u64..end as u64)
                .map(|id| mix64(id ^ dimension as u64))
                .collect::<Vec<_>>();
            let batch = RecordBatch::try_new(
                Arc::clone(&dimension_schema),
                vec![
                    Arc::new(UInt64Array::from(ids)),
                    Arc::new(UInt64Array::from(markers)),
                    Arc::new(UInt64Array::from(payload)),
                ],
            )?;
            let partition = (start / options.batch_size) % options.partitions;
            partitions[partition].push(batch);
            start = end;
        }
        tables.push((
            format!("dim{dimension}"),
            Arc::new(MemTable::try_new(
                Arc::clone(&dimension_schema),
                partitions,
            )?),
        ));
    }
    Ok(tables)
}

fn star_query(options: &Options) -> String {
    let mut sql = "SELECT count(*) AS matches FROM fact f".to_string();
    for dimension in 0..options.dimensions {
        sql.push_str(&format!(
            " JOIN dim{dimension} d{dimension} ON f.k{dimension} = d{dimension}.id"
        ));
    }
    sql.push_str(" WHERE ");
    for dimension in 0..options.dimensions {
        if dimension > 0 {
            sql.push_str(" AND ");
        }
        sql.push_str(&format!(
            "d{dimension}.marker < {}",
            options.selectivity_bps
        ));
    }
    sql
}

fn build_balanced(options: &Options) -> Result<Vec<(String, Arc<dyn TableProvider>)>> {
    assert!(options.dimensions.is_power_of_two());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("marker", DataType::UInt64, false),
        Field::new("payload", DataType::UInt64, false),
    ]));
    let mut tables: Vec<(String, Arc<dyn TableProvider>)> = Vec::with_capacity(options.dimensions);
    for relation in 0..options.dimensions {
        let mut partitions = vec![vec![]; options.partitions];
        let mut start = 0;
        while start < options.fact_rows {
            let end = (start + options.batch_size).min(options.fact_rows);
            let ids = (start as u64..end as u64).collect::<Vec<_>>();
            let markers = (start as u64..end as u64)
                .map(|id| mix64(id ^ ((relation as u64 + 31) * 0x9e37_79b9)) % 10_000)
                .collect::<Vec<_>>();
            let payload = (start as u64..end as u64)
                .map(|id| mix64(id ^ relation as u64))
                .collect::<Vec<_>>();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(UInt64Array::from(ids)),
                    Arc::new(UInt64Array::from(markers)),
                    Arc::new(UInt64Array::from(payload)),
                ],
            )?;
            let partition = (start / options.batch_size) % options.partitions;
            partitions[partition].push(batch);
            start = end;
        }
        tables.push((
            format!("rel{relation}"),
            Arc::new(MemTable::try_new(Arc::clone(&schema), partitions)?),
        ));
    }
    Ok(tables)
}

fn balanced_query(options: &Options) -> String {
    fn join_tree(relations: &[usize]) -> (String, usize) {
        if let [relation] = relations {
            return (format!("rel{relation} r{relation}"), *relation);
        }
        let middle = relations.len() / 2;
        let (left, left_key) = join_tree(&relations[..middle]);
        let (right, right_key) = join_tree(&relations[middle..]);
        (
            format!("({left} JOIN {right} ON r{left_key}.id = r{right_key}.id)"),
            left_key,
        )
    }

    let relations = (0..options.dimensions).collect::<Vec<_>>();
    let (tree, _) = join_tree(&relations);
    format!(
        "SELECT count(*) AS matches FROM {tree} WHERE r0.marker < {} AND r{}.marker < {}",
        options.selectivity_bps,
        options.dimensions - 1,
        options.selectivity_bps
    )
}

fn count_named(plan: &Arc<dyn ExecutionPlan>, needle: &str) -> usize {
    usize::from(plan.name().contains(needle))
        + plan
            .children()
            .into_iter()
            .map(|child| count_named(child, needle))
            .sum::<usize>()
}

fn median_ms<F>(runs: &[Measurement], field: F) -> f64
where
    F: Fn(&Measurement) -> Duration,
{
    let mut values = runs
        .iter()
        .map(|run| field(run).as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    values.sort_unstable_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    values[values.len() / 2]
}

fn parse<T>(flag: &str, value: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .unwrap_or_else(|error| panic!("invalid value for {flag}: {value}: {error}"))
}

#[inline]
fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
