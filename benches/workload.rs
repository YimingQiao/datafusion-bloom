use std::cmp::Ordering;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::hash_utils::{RandomState, create_hashes};
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::physical_plan::parquet::ParquetAccessPlan;
use datafusion::datasource::physical_plan::{FileScanConfig, FileSource, ParquetSource};
use datafusion::datasource::source::DataSourceExec;
use datafusion::datasource::{MemTable, TableProvider};
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::{collect, collect_partitioned, displayable};
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_bloom::{BloomConfig, install_bloom};
use tokio::runtime::Builder as RuntimeBuilder;

const JOB_TABLES: &[&str] = &[
    "aka_name",
    "aka_title",
    "cast_info",
    "char_name",
    "comp_cast_type",
    "company_name",
    "company_type",
    "complete_cast",
    "info_type",
    "keyword",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "name",
    "person_info",
    "role_type",
    "title",
];

const TPCH_TABLES: &[&str] = &[
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Workload {
    Job,
    CebImdb,
    Tpch,
}

impl Workload {
    fn parse(value: &str) -> Self {
        match value {
            "job" => Self::Job,
            "ceb-imdb" | "ceb_imdb" | "ceb" => Self::CebImdb,
            "tpch" => Self::Tpch,
            _ => panic!("--workload must be job, ceb-imdb, or tpch, got {value}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Job => "job",
            Self::CebImdb => "ceb-imdb",
            Self::Tpch => "tpch",
        }
    }

    fn tables(self) -> &'static [&'static str] {
        match self {
            Self::Job | Self::CebImdb => JOB_TABLES,
            Self::Tpch => TPCH_TABLES,
        }
    }
}

#[derive(Debug, Clone)]
struct Options {
    workload: Workload,
    data_dir: PathBuf,
    query_dir: PathBuf,
    selected_queries: Option<HashSet<String>>,
    scale_factor: usize,
    threads: usize,
    batch_size: Option<usize>,
    excitation_threshold: f64,
    warmups: usize,
    runs: usize,
    show_plan: bool,
    show_metrics: bool,
    plan_only: bool,
    handoff_audit: bool,
    baseline_only: bool,
    bloom_only: bool,
    row_locations: bool,
    log_transfer: bool,
    fresh_context_per_query: bool,
    parquet_pushdown: bool,
    post_scan_membership: bool,
    predicate_cache_size: Option<usize>,
    preload_memory: bool,
    reoptimize: bool,
    utf8view: bool,
}

impl Options {
    fn parse() -> Self {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut workload = Workload::Job;
        let mut data_dir = None;
        let mut query_dir = None;
        let mut selected_queries = None;
        let mut scale_factor = 10;
        let mut threads = 1;
        let mut batch_size = None;
        let mut excitation_threshold = 1.0;
        let mut warmups = 1;
        let mut runs = 5;
        let mut show_plan = false;
        let mut show_metrics = false;
        let mut plan_only = false;
        let mut handoff_audit = false;
        let mut baseline_only = false;
        let mut bloom_only = false;
        let mut row_locations = false;
        let mut log_transfer = false;
        let mut fresh_context_per_query = false;
        let mut parquet_pushdown = false;
        let mut post_scan_membership = false;
        let mut predicate_cache_size = None;
        let mut preload_memory = false;
        let mut reoptimize = false;
        let mut utf8view = false;

        let mut arguments = env::args().skip(1);
        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                // Cargo adds this marker when launching a custom benchmark.
                "--bench" => {}
                "--show-plan" => show_plan = true,
                "--show-metrics" => show_metrics = true,
                "--plan-only" => plan_only = true,
                "--handoff-audit" => handoff_audit = true,
                "--baseline-only" => baseline_only = true,
                "--bloom-only" => bloom_only = true,
                "--row-locations" => row_locations = true,
                "--log-transfer" => log_transfer = true,
                "--fresh-context-per-query" => fresh_context_per_query = true,
                "--parquet-pushdown" => parquet_pushdown = true,
                "--post-scan-membership" => post_scan_membership = true,
                "--predicate-cache-size" => {
                    predicate_cache_size = Some(parse(&flag, &next_value(&mut arguments, &flag)))
                }
                "--preload-memory" => preload_memory = true,
                "--reoptimize" => reoptimize = true,
                "--utf8view" => utf8view = true,
                "--workload" => workload = Workload::parse(&next_value(&mut arguments, &flag)),
                "--data-dir" => data_dir = Some(PathBuf::from(next_value(&mut arguments, &flag))),
                "--query-dir" => query_dir = Some(PathBuf::from(next_value(&mut arguments, &flag))),
                "--queries" => {
                    let value = next_value(&mut arguments, &flag);
                    selected_queries = if value == "all" {
                        None
                    } else {
                        Some(
                            value
                                .split(',')
                                .map(|name| name.trim().trim_end_matches(".sql").to_string())
                                .filter(|name| !name.is_empty())
                                .collect(),
                        )
                    };
                }
                "--scale-factor" => scale_factor = parse(&flag, &next_value(&mut arguments, &flag)),
                "--threads" => threads = parse(&flag, &next_value(&mut arguments, &flag)),
                "--batch-size" => {
                    batch_size = Some(parse(&flag, &next_value(&mut arguments, &flag)))
                }
                "--excitation-threshold" => {
                    excitation_threshold = parse(&flag, &next_value(&mut arguments, &flag))
                }
                "--warmups" => warmups = parse(&flag, &next_value(&mut arguments, &flag)),
                "--runs" => runs = parse(&flag, &next_value(&mut arguments, &flag)),
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => panic!("unknown argument: {flag}; use --help for usage"),
            }
        }

        assert!(scale_factor > 0, "scale factor must be greater than zero");
        assert!(threads > 0, "threads must be greater than zero");
        assert!(
            batch_size.is_none_or(|size| size > 0),
            "batch size must be greater than zero"
        );
        assert!(
            excitation_threshold > 0.0 && excitation_threshold <= 1.0,
            "excitation threshold must be greater than zero and at most one"
        );
        assert!(runs > 0, "runs must be greater than zero");
        assert!(
            usize::from(handoff_audit) + usize::from(baseline_only) + usize::from(bloom_only) <= 1,
            "--handoff-audit, --baseline-only, and --bloom-only are mutually exclusive"
        );
        assert!(
            !preload_memory || handoff_audit || bloom_only || plan_only,
            "--preload-memory is diagnostic and cannot report a Baseline/Bloom speedup; use \
             --bloom-only, --handoff-audit, or --plan-only"
        );

        let data_dir = data_dir.unwrap_or_else(|| match workload {
            Workload::Job | Workload::CebImdb => manifest.join("benchmark_data/job/parquet"),
            Workload::Tpch => {
                manifest.join(format!("benchmark_data/tpch/sf{scale_factor}/parquet"))
            }
        });
        let query_dir = query_dir.unwrap_or_else(|| match workload {
            Workload::CebImdb => manifest.join("benchmark_data/ceb-imdb/queries"),
            Workload::Job | Workload::Tpch => manifest
                .join("benchmark")
                .join(workload.name())
                .join("queries"),
        });

        Self {
            workload,
            data_dir,
            query_dir,
            selected_queries,
            scale_factor,
            threads,
            batch_size,
            excitation_threshold,
            warmups,
            runs,
            show_plan,
            show_metrics,
            plan_only,
            handoff_audit,
            baseline_only,
            bloom_only,
            row_locations,
            log_transfer,
            fresh_context_per_query,
            parquet_pushdown,
            post_scan_membership,
            predicate_cache_size,
            preload_memory,
            reoptimize,
            utf8view,
        }
    }
}

#[derive(Debug)]
struct PreloadedTable {
    name: &'static str,
    schema: datafusion::arrow::datatypes::SchemaRef,
    partitions: Vec<Vec<RecordBatch>>,
}

#[derive(Debug)]
struct PreloadedData {
    tables: Vec<PreloadedTable>,
    elapsed: Duration,
    rows: usize,
    physical_bytes: usize,
    batch_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Fingerprint {
    rows: usize,
    sum: u64,
    mixed_sum: u64,
    xor: u64,
}

#[derive(Debug)]
struct Measurement {
    planning: Duration,
    execution: Duration,
    fingerprint: Fingerprint,
    full_rows_handoffs: usize,
    row_location_handoffs: usize,
    direct_handoffs: usize,
    metrics: Option<String>,
}

impl Measurement {
    fn total(&self) -> Duration {
        self.planning + self.execution
    }
}

#[derive(Debug)]
struct QuerySummary {
    name: String,
    baseline_plan_ms: f64,
    baseline_exec_ms: f64,
    baseline_total_ms: f64,
    bloom_plan_ms: f64,
    bloom_exec_ms: f64,
    bloom_total_ms: f64,
    full_rows_handoffs: usize,
    row_location_handoffs: usize,
    direct_handoffs: usize,
    rows: usize,
}

fn main() -> Result<()> {
    let options = Options::parse();
    let mut runtime_builder = if options.threads == 1 {
        RuntimeBuilder::new_current_thread()
    } else {
        let mut builder = RuntimeBuilder::new_multi_thread();
        builder.worker_threads(options.threads);
        builder
    };
    runtime_builder.enable_all().build()?.block_on(run(options))
}

async fn run(options: Options) -> Result<()> {
    validate_layout(&options)?;
    let queries = load_queries(&options)?;
    let preloaded = preload_data(&options).await?;
    if let Some(data) = &preloaded {
        println!(
            "Preloaded Arrow input (outside query timing): elapsed_ms={:.3} tables={} rows={} physical_bytes={} batch_size={}",
            data.elapsed.as_secs_f64() * 1000.0,
            data.tables.len(),
            data.rows,
            data.physical_bytes,
            data.batch_size,
        );
    }

    if options.handoff_audit {
        let bloom = make_context(&options, true, preloaded.as_ref()).await?;
        println!("Bloom handoff audit (complete output; no baseline timing)");
        println!("query\ttotal_ms\tfull_rows\trow_locations\tdirect\trows");
        for query in &queries {
            let measurement = measure(&bloom, &query.sql, false).await?;
            println!(
                "{}\t{:.3}\t{}\t{}\t{}\t{}",
                query.name,
                measurement.total().as_secs_f64() * 1000.0,
                measurement.full_rows_handoffs,
                measurement.row_location_handoffs,
                measurement.direct_handoffs,
                measurement.fingerprint.rows,
            );
        }
        return Ok(());
    }

    if options.baseline_only {
        return run_baseline_only(&options, &queries, preloaded.as_ref()).await;
    }

    if options.bloom_only {
        return run_bloom_only(&options, &queries, preloaded.as_ref()).await;
    }

    let baseline = make_context(&options, false, preloaded.as_ref()).await?;
    let bloom = make_context(&options, true, preloaded.as_ref()).await?;

    if options.plan_only {
        for query in &queries {
            let baseline_plan = baseline
                .sql(&query.sql)
                .await?
                .create_physical_plan()
                .await?;
            let bloom_plan = bloom.sql(&query.sql).await?.create_physical_plan().await?;
            let baseline_features = PlanFeatures::collect(baseline_plan.as_ref())?;
            let bloom_features = PlanFeatures::collect(bloom_plan.as_ref())?;
            println!(
                "\n[{} DataFusion plan]\n{}\n{}",
                query.name,
                baseline_features,
                displayable(baseline_plan.as_ref()).indent(true)
            );
            println!(
                "\n[{} Bloom plan]\n{}\n{}",
                query.name,
                bloom_features,
                displayable(bloom_plan.as_ref()).indent(true)
            );
        }
        return Ok(());
    }

    println!(
        "Bloom {} end-to-end benchmark",
        options.workload.name().to_uppercase()
    );
    println!(
        "data={} source={} queries={} selected={} threads={} batch_size={} string_type={} excitation_threshold={} warmups={} runs={} scale_factor={} row_locations={} reoptimize={} parquet_pushdown={} predicate_cache_size={:?} membership={} context={}",
        options.data_dir.display(),
        source_name(&options),
        options.query_dir.display(),
        queries.len(),
        options.threads,
        effective_batch_size(&options),
        string_type_name(&options),
        options.excitation_threshold,
        options.warmups,
        options.runs,
        options.scale_factor,
        options.row_locations,
        options.reoptimize,
        options.parquet_pushdown,
        options.predicate_cache_size,
        if options.post_scan_membership {
            "post-scan"
        } else {
            "reader"
        },
        if options.fresh_context_per_query {
            "fresh-per-query"
        } else {
            "shared-workload"
        },
    );
    println!(
        "query\tbaseline_plan_ms\tbaseline_exec_ms\tbaseline_total_ms\tbloom_plan_ms\tbloom_exec_ms\tbloom_total_ms\tspeedup\tfull_rows\trow_locations\tdirect\trows"
    );

    let mut summaries = Vec::with_capacity(queries.len());
    for (query_index, query) in queries.iter().enumerate() {
        // The normal workload mode models a long-lived service and permits
        // query-independent metadata/sample reuse. This optional mode gives
        // every SQL statement a newly registered pair of sessions so cold
        // single-query latency can be reported separately. Context creation
        // remains outside the query timer in both modes.
        let fresh_contexts = if options.fresh_context_per_query {
            Some((
                make_context(&options, false, preloaded.as_ref()).await?,
                make_context(&options, true, preloaded.as_ref()).await?,
            ))
        } else {
            None
        };
        let (query_baseline, query_bloom) = fresh_contexts
            .as_ref()
            .map(|(baseline, bloom)| (baseline, bloom))
            .unwrap_or((&baseline, &bloom));

        for warmup in 0..options.warmups {
            let baseline_first = (query_index + warmup) % 2 == 0;
            let (left, right) = execute_pair(
                query_baseline,
                query_bloom,
                &query.sql,
                baseline_first,
                options.show_metrics,
            )
            .await?;
            verify(&query.name, &left, &right)?;
        }

        let mut baseline_runs = Vec::with_capacity(options.runs);
        let mut bloom_runs = Vec::with_capacity(options.runs);
        for run in 0..options.runs {
            let baseline_first = (query_index + run) % 2 == 0;
            let (left, right) = execute_pair(
                query_baseline,
                query_bloom,
                &query.sql,
                baseline_first,
                options.show_metrics,
            )
            .await?;
            verify(&query.name, &left, &right)?;
            baseline_runs.push(left);
            bloom_runs.push(right);
        }

        let summary = QuerySummary {
            name: query.name.clone(),
            baseline_plan_ms: median_ms(&baseline_runs, |run| run.planning),
            baseline_exec_ms: median_ms(&baseline_runs, |run| run.execution),
            baseline_total_ms: median_ms(&baseline_runs, Measurement::total),
            bloom_plan_ms: median_ms(&bloom_runs, |run| run.planning),
            bloom_exec_ms: median_ms(&bloom_runs, |run| run.execution),
            bloom_total_ms: median_ms(&bloom_runs, Measurement::total),
            full_rows_handoffs: bloom_runs
                .iter()
                .map(|run| run.full_rows_handoffs)
                .max()
                .unwrap_or(0),
            row_location_handoffs: bloom_runs
                .iter()
                .map(|run| run.row_location_handoffs)
                .max()
                .unwrap_or(0),
            direct_handoffs: bloom_runs
                .iter()
                .map(|run| run.direct_handoffs)
                .max()
                .unwrap_or(0),
            rows: bloom_runs[0].fingerprint.rows,
        };
        print_summary(&summary);

        if options.show_metrics {
            println!(
                "\n[{} DataFusion metrics]\n{}",
                query.name,
                baseline_runs
                    .last()
                    .and_then(|run| run.metrics.as_deref())
                    .unwrap_or("metrics unavailable")
            );
            println!(
                "\n[{} Bloom metrics]\n{}",
                query.name,
                bloom_runs
                    .last()
                    .and_then(|run| run.metrics.as_deref())
                    .unwrap_or("metrics unavailable")
            );
        }

        if options.show_plan {
            print_plans(query_baseline, query_bloom, query).await?;
        }
        summaries.push(summary);
    }

    print_totals(&summaries);
    Ok(())
}

async fn run_baseline_only(
    options: &Options,
    queries: &[Query],
    preloaded: Option<&PreloadedData>,
) -> Result<()> {
    let shared = make_context(options, false, preloaded).await?;
    println!(
        "DataFusion {} end-to-end benchmark (baseline only)",
        options.workload.name().to_uppercase()
    );
    println!(
        "data={} source={} queries={} selected={} threads={} batch_size={} string_type={} warmups={} runs={} context={}",
        options.data_dir.display(),
        source_name(options),
        options.query_dir.display(),
        queries.len(),
        options.threads,
        effective_batch_size(options),
        string_type_name(options),
        options.warmups,
        options.runs,
        if options.fresh_context_per_query {
            "fresh-per-query"
        } else {
            "shared-workload"
        },
    );
    println!("query\tbaseline_plan_ms\tbaseline_exec_ms\tbaseline_total_ms\trows");

    let mut total_ms = 0.0;
    for query in queries {
        let fresh;
        let baseline = if options.fresh_context_per_query {
            fresh = make_context(options, false, preloaded).await?;
            &fresh
        } else {
            &shared
        };

        let mut expected = None;
        for _ in 0..options.warmups {
            let measurement = measure(baseline, &query.sql, false).await?;
            verify_bloom_run(&query.name, &mut expected, &measurement)?;
        }

        let mut runs = Vec::with_capacity(options.runs);
        for _ in 0..options.runs {
            let measurement = measure(baseline, &query.sql, options.show_metrics).await?;
            verify_bloom_run(&query.name, &mut expected, &measurement)?;
            runs.push(measurement);
        }

        let planning_ms = median_ms(&runs, |run| run.planning);
        let execution_ms = median_ms(&runs, |run| run.execution);
        let query_total_ms = median_ms(&runs, Measurement::total);
        total_ms += query_total_ms;
        println!(
            "{}\t{planning_ms:.3}\t{execution_ms:.3}\t{query_total_ms:.3}\t{}",
            query.name, runs[0].fingerprint.rows
        );
        if options.show_metrics {
            println!(
                "\n[{} DataFusion metrics]\n{}",
                query.name,
                runs.last()
                    .and_then(|run| run.metrics.as_deref())
                    .unwrap_or("metrics unavailable")
            );
        }
    }
    println!("\nTOTAL\tbaseline_median_sum_ms={total_ms:.3}");
    Ok(())
}

async fn run_bloom_only(
    options: &Options,
    queries: &[Query],
    preloaded: Option<&PreloadedData>,
) -> Result<()> {
    let shared = make_context(options, true, preloaded).await?;
    println!(
        "Bloom {} end-to-end benchmark (Bloom only)",
        options.workload.name().to_uppercase()
    );
    println!(
        "data={} source={} queries={} selected={} threads={} batch_size={} string_type={} excitation_threshold={} warmups={} runs={} reoptimize={} predicate_cache_size={:?} membership={} context={}",
        options.data_dir.display(),
        source_name(options),
        options.query_dir.display(),
        queries.len(),
        options.threads,
        effective_batch_size(options),
        string_type_name(options),
        options.excitation_threshold,
        options.warmups,
        options.runs,
        options.reoptimize,
        options.predicate_cache_size,
        if options.post_scan_membership {
            "post-scan"
        } else {
            "reader"
        },
        if options.fresh_context_per_query {
            "fresh-per-query"
        } else {
            "shared-workload"
        },
    );
    println!(
        "query\tbloom_plan_ms\tbloom_exec_ms\tbloom_total_ms\tfull_rows\trow_locations\tdirect\trows"
    );

    let mut total_ms = 0.0;
    for query in queries {
        let fresh;
        let bloom = if options.fresh_context_per_query {
            fresh = make_context(options, true, preloaded).await?;
            &fresh
        } else {
            &shared
        };

        let mut expected = None;
        for _ in 0..options.warmups {
            let measurement = measure(bloom, &query.sql, false).await?;
            verify_bloom_run(&query.name, &mut expected, &measurement)?;
        }

        let mut runs = Vec::with_capacity(options.runs);
        for _ in 0..options.runs {
            let measurement = measure(bloom, &query.sql, options.show_metrics).await?;
            verify_bloom_run(&query.name, &mut expected, &measurement)?;
            runs.push(measurement);
        }

        let planning_ms = median_ms(&runs, |run| run.planning);
        let execution_ms = median_ms(&runs, |run| run.execution);
        let query_total_ms = median_ms(&runs, Measurement::total);
        let full_rows = runs
            .iter()
            .map(|run| run.full_rows_handoffs)
            .max()
            .unwrap_or(0);
        let row_locations = runs
            .iter()
            .map(|run| run.row_location_handoffs)
            .max()
            .unwrap_or(0);
        let direct = runs
            .iter()
            .map(|run| run.direct_handoffs)
            .max()
            .unwrap_or(0);
        println!(
            "{}\t{planning_ms:.3}\t{execution_ms:.3}\t{query_total_ms:.3}\t{full_rows}\t{row_locations}\t{direct}\t{}",
            query.name, runs[0].fingerprint.rows
        );
        total_ms += query_total_ms;

        if options.show_metrics {
            println!(
                "\n[{} Bloom metrics]\n{}",
                query.name,
                runs.last()
                    .and_then(|run| run.metrics.as_deref())
                    .unwrap_or("metrics unavailable")
            );
        }
        if options.show_plan {
            let plan = bloom.sql(&query.sql).await?.create_physical_plan().await?;
            println!(
                "\n[{} Bloom plan]\n{}",
                query.name,
                displayable(plan.as_ref()).indent(false)
            );
        }
    }
    println!("\nTOTAL\tbloom_median_sum_ms={total_ms:.3}");
    Ok(())
}

fn verify_bloom_run(
    query: &str,
    expected: &mut Option<Fingerprint>,
    measurement: &Measurement,
) -> Result<()> {
    if let Some(expected) = expected {
        if expected != &measurement.fingerprint {
            return Err(DataFusionError::Execution(format!(
                "unstable Bloom result for {query}: expected={expected:?}, actual={:?}",
                measurement.fingerprint
            )));
        }
    } else {
        *expected = Some(measurement.fingerprint.clone());
    }
    Ok(())
}

#[derive(Debug, Default)]
struct PlanFeatures {
    joins: usize,
    sources: usize,
    unfiltered_sources: usize,
    sum_join_rows: usize,
    max_join_rows: usize,
    max_source_rows: usize,
    max_unfiltered_source_rows: usize,
}

impl PlanFeatures {
    fn collect(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> Result<Self> {
        let mut features = Self::default();
        features.visit(plan)?;
        Ok(features)
    }

    fn visit(&mut self, plan: &dyn datafusion::physical_plan::ExecutionPlan) -> Result<()> {
        let rows = plan
            .partition_statistics(None)?
            .num_rows
            .get_value()
            .copied()
            .unwrap_or(0);
        if plan.downcast_ref::<HashJoinExec>().is_some() {
            self.joins += 1;
            self.sum_join_rows = self.sum_join_rows.saturating_add(rows);
            self.max_join_rows = self.max_join_rows.max(rows);
        }
        if let Some(source) = plan.downcast_ref::<DataSourceExec>() {
            self.sources += 1;
            self.max_source_rows = self.max_source_rows.max(rows);
            let filtered = source
                .data_source()
                .downcast_ref::<FileScanConfig>()
                .and_then(|config| config.file_source().downcast_ref::<ParquetSource>())
                .and_then(ParquetSource::filter)
                .is_some();
            if !filtered {
                self.unfiltered_sources += 1;
                self.max_unfiltered_source_rows = self.max_unfiltered_source_rows.max(rows);
            }
        }
        for child in plan.children() {
            self.visit(child.as_ref())?;
        }
        Ok(())
    }
}

impl std::fmt::Display for PlanFeatures {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "features: joins={} sources={} unfiltered_sources={} sum_join_rows={} max_join_rows={} max_source_rows={} max_unfiltered_source_rows={}",
            self.joins,
            self.sources,
            self.unfiltered_sources,
            self.sum_join_rows,
            self.max_join_rows,
            self.max_source_rows,
            self.max_unfiltered_source_rows
        )
    }
}

fn benchmark_session_config(options: &Options) -> SessionConfig {
    let mut config = SessionConfig::new().with_target_partitions(options.threads);
    // Arrow's byte-view take path can retain and repeatedly clone a growing
    // backing-buffer graph across high-fanout joins. Until that upstream path
    // is fixed, the release benchmark uses ordinary Utf8 for both stock
    // DataFusion and Bloom. `--utf8view` restores DataFusion 54.1's native
    // default for an explicit sensitivity comparison.
    config.options_mut().sql_parser.map_string_types_to_utf8view = options.utf8view;
    config
        .options_mut()
        .execution
        .parquet
        .schema_force_view_types = options.utf8view;
    if let Some(batch_size) = options.batch_size {
        config = config.with_batch_size(batch_size);
    }
    if options.parquet_pushdown {
        config.options_mut().execution.parquet.pushdown_filters = true;
        config.options_mut().execution.parquet.reorder_filters = true;
    }
    if let Some(size) = options.predicate_cache_size {
        config
            .options_mut()
            .execution
            .parquet
            .max_predicate_cache_size = Some(size);
    }
    config
}

fn effective_batch_size(options: &Options) -> usize {
    options
        .batch_size
        .unwrap_or_else(|| SessionConfig::new().batch_size())
}

async fn preload_data(options: &Options) -> Result<Option<PreloadedData>> {
    if !options.preload_memory {
        return Ok(None);
    }

    let started = Instant::now();
    let loader_config = benchmark_session_config(options);
    let batch_size = loader_config.batch_size();
    let loader = SessionContext::new_with_config(loader_config);
    for table in options.workload.tables() {
        let path = table_path(&options.data_dir, table);
        loader
            .register_parquet(*table, path_string(&path), ParquetReadOptions::default())
            .await?;
    }

    let mut tables = Vec::with_capacity(options.workload.tables().len());
    let mut rows = 0usize;
    let mut physical_bytes = 0usize;
    for &name in options.workload.tables() {
        let plan = loader.table(name).await?.create_physical_plan().await?;
        let schema = plan.schema();
        let partitions = collect_partitioned(plan, loader.task_ctx()).await?;
        rows = rows.saturating_add(
            partitions
                .iter()
                .flatten()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
        );
        physical_bytes = physical_bytes.saturating_add(
            partitions
                .iter()
                .flatten()
                .flat_map(RecordBatch::columns)
                .map(|array| array.get_array_memory_size())
                .sum::<usize>(),
        );
        tables.push(PreloadedTable {
            name,
            schema,
            partitions,
        });
    }
    Ok(Some(PreloadedData {
        tables,
        elapsed: started.elapsed(),
        rows,
        physical_bytes,
        batch_size,
    }))
}

async fn make_context(
    options: &Options,
    bloom: bool,
    preloaded: Option<&PreloadedData>,
) -> Result<SessionContext> {
    let config = benchmark_session_config(options);
    let mut state = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .build();
    if bloom {
        let mut bloom_config = BloomConfig::default().with_all_bounded_sources();
        bloom_config.excitation_threshold = options.excitation_threshold;
        bloom_config.reoptimize = options.reoptimize;
        if options.row_locations {
            bloom_config = bloom_config.with_row_locations();
        }
        if options.post_scan_membership {
            bloom_config = bloom_config.with_post_scan_membership();
        }
        if options.log_transfer {
            bloom_config = bloom_config.with_transfer_logging();
        }
        state = install_bloom(state, bloom_config)?;
    }
    let context = SessionContext::new_with_state(state);
    if let Some(data) = preloaded {
        for table in &data.tables {
            let provider: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
                Arc::clone(&table.schema),
                table.partitions.clone(),
            )?);
            context.register_table(table.name, provider)?;
        }
    } else {
        for table in options.workload.tables() {
            let path = table_path(&options.data_dir, table);
            context
                .register_parquet(*table, path_string(&path), ParquetReadOptions::default())
                .await?;
        }
    }
    Ok(context)
}

async fn execute_pair(
    baseline: &SessionContext,
    bloom: &SessionContext,
    sql: &str,
    baseline_first: bool,
    show_metrics: bool,
) -> Result<(Measurement, Measurement)> {
    if baseline_first {
        let left = measure(baseline, sql, show_metrics).await?;
        let right = measure(bloom, sql, show_metrics).await?;
        Ok((left, right))
    } else {
        let right = measure(bloom, sql, show_metrics).await?;
        let left = measure(baseline, sql, show_metrics).await?;
        Ok((left, right))
    }
}

async fn measure(context: &SessionContext, sql: &str, show_metrics: bool) -> Result<Measurement> {
    let started = Instant::now();
    let frame = context.sql(sql).await?;
    let plan = frame.create_physical_plan().await?;
    let planned = Instant::now();
    let batches = collect(Arc::clone(&plan), context.task_ctx()).await?;
    let finished = Instant::now();

    // Fingerprinting and plan diagnostics are deliberately outside the timed
    // interval. `collect` has already materialized the complete query output.
    let fingerprint = fingerprint(&batches)?;
    let formatted_plan = displayable(plan.as_ref()).indent(false).to_string();
    let full_rows_handoffs = formatted_plan.matches("BloomCollection").count();
    let row_location_handoffs = count_row_selected_sources(&plan);
    let direct_handoffs = formatted_plan.matches("BloomScanBoundaryExec").count();
    let metrics = show_metrics.then(|| {
        DisplayableExecutionPlan::with_metrics(plan.as_ref())
            .indent(false)
            .to_string()
    });
    Ok(Measurement {
        planning: planned.duration_since(started),
        execution: finished.duration_since(planned),
        fingerprint,
        full_rows_handoffs,
        row_location_handoffs,
        direct_handoffs,
        metrics,
    })
}

fn fingerprint(batches: &[RecordBatch]) -> Result<Fingerprint> {
    let random_state = RandomState::default();
    let mut result = Fingerprint {
        rows: 0,
        sum: 0,
        mixed_sum: 0,
        xor: 0,
    };
    for batch in batches {
        let mut hashes = vec![0_u64; batch.num_rows()];
        create_hashes(batch.columns(), &random_state, &mut hashes)?;
        result.rows += hashes.len();
        for hash in hashes {
            result.sum = result.sum.wrapping_add(hash);
            let mixed = mix64(hash ^ 0x9e37_79b9_7f4a_7c15);
            result.mixed_sum = result.mixed_sum.wrapping_add(mixed);
            result.xor ^= mixed.rotate_left(23);
        }
    }
    Ok(result)
}

fn count_row_selected_sources(plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>) -> usize {
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

fn verify(query: &str, baseline: &Measurement, bloom: &Measurement) -> Result<()> {
    if baseline.fingerprint != bloom.fingerprint {
        return Err(DataFusionError::Execution(format!(
            "result mismatch for {query}: baseline={:?}, bloom={:?}",
            baseline.fingerprint, bloom.fingerprint
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct Query {
    name: String,
    sql: String,
}

fn load_queries(options: &Options) -> Result<Vec<Query>> {
    let mut paths = vec![];
    collect_sql_paths(&options.query_dir, &mut paths)?;
    paths.sort_by(|left, right| {
        query_key(left)
            .cmp(&query_key(right))
            .then_with(|| left.cmp(right))
    });

    let mut queries = Vec::new();
    let mut found = HashSet::new();
    for path in paths {
        let name = query_name(&options.query_dir, &path)?;
        if options
            .selected_queries
            .as_ref()
            .is_some_and(|selected| !selected.contains(&name))
        {
            continue;
        }
        let sql = fs::read_to_string(&path)?;
        if !found.insert(name.clone()) {
            return Err(DataFusionError::Execution(format!(
                "duplicate normalized query name: {name}"
            )));
        }
        queries.push(Query { name, sql });
    }

    if let Some(selected) = &options.selected_queries {
        let mut missing = selected.difference(&found).cloned().collect::<Vec<_>>();
        missing.sort();
        if !missing.is_empty() {
            return Err(DataFusionError::Execution(format!(
                "selected queries not found: {}",
                missing.join(",")
            )));
        }
    }
    if queries.is_empty() {
        return Err(DataFusionError::Execution(
            "no queries selected".to_string(),
        ));
    }
    Ok(queries)
}

fn collect_sql_paths(directory: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_sql_paths(&path, output)?;
        } else if path.extension().is_some_and(|extension| extension == "sql") {
            output.push(path);
        }
    }
    Ok(())
}

fn query_name(root: &Path, path: &Path) -> Result<String> {
    let relative = path.strip_prefix(root).map_err(|error| {
        DataFusionError::Execution(format!(
            "query path {} is outside {}: {error}",
            path.display(),
            root.display()
        ))
    })?;
    let without_extension = relative.with_extension("");
    let components = without_extension
        .iter()
        .map(|component| component.to_str())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            DataFusionError::Execution(format!("invalid query path: {}", path.display()))
        })?;
    Ok(components.join("__"))
}

fn validate_layout(options: &Options) -> Result<()> {
    if !options.query_dir.is_dir() {
        return Err(DataFusionError::Execution(format!(
            "query directory does not exist: {}",
            options.query_dir.display()
        )));
    }
    for table in options.workload.tables() {
        let path = table_path(&options.data_dir, table);
        if !path.exists() {
            return Err(DataFusionError::Execution(format!(
                "table data does not exist: {}; run the workload preparation script first",
                path.display()
            )));
        }
    }
    Ok(())
}

fn table_path(data_dir: &Path, table: &str) -> PathBuf {
    let directory = data_dir.join(table);
    if directory.exists() {
        directory
    } else {
        data_dir.join(format!("{table}.parquet"))
    }
}

async fn print_plans(
    baseline: &SessionContext,
    bloom: &SessionContext,
    query: &Query,
) -> Result<()> {
    let baseline_plan = baseline
        .sql(&query.sql)
        .await?
        .create_physical_plan()
        .await?;
    let bloom_plan = bloom.sql(&query.sql).await?.create_physical_plan().await?;
    println!(
        "\n[{} DataFusion plan]\n{}",
        query.name,
        displayable(baseline_plan.as_ref()).indent(false)
    );
    println!(
        "\n[{} Bloom plan]\n{}",
        query.name,
        displayable(bloom_plan.as_ref()).indent(false)
    );
    Ok(())
}

fn print_summary(summary: &QuerySummary) {
    println!(
        "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}",
        summary.name,
        summary.baseline_plan_ms,
        summary.baseline_exec_ms,
        summary.baseline_total_ms,
        summary.bloom_plan_ms,
        summary.bloom_exec_ms,
        summary.bloom_total_ms,
        summary.baseline_total_ms / summary.bloom_total_ms,
        summary.full_rows_handoffs,
        summary.row_location_handoffs,
        summary.direct_handoffs,
        summary.rows,
    );
}

fn print_totals(summaries: &[QuerySummary]) {
    let baseline_total = summaries
        .iter()
        .map(|query| query.baseline_total_ms)
        .sum::<f64>();
    let bloom_total = summaries
        .iter()
        .map(|query| query.bloom_total_ms)
        .sum::<f64>();
    let geometric_mean = (summaries
        .iter()
        .map(|query| (query.baseline_total_ms / query.bloom_total_ms).ln())
        .sum::<f64>()
        / summaries.len() as f64)
        .exp();
    let faster = summaries
        .iter()
        .filter(|query| query.bloom_total_ms < query.baseline_total_ms)
        .count();
    println!();
    println!(
        "TOTAL\tbaseline_median_sum_ms={baseline_total:.3}\tbloom_median_sum_ms={bloom_total:.3}\tworkload_speedup={:.3}x\tgeomean_speedup={geometric_mean:.3}x\tfaster={faster}/{}",
        baseline_total / bloom_total,
        summaries.len()
    );
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

fn query_key(path: &Path) -> (usize, String) {
    let stem = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let digits = stem
        .trim_start_matches(|character: char| !character.is_ascii_digit())
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    (digits.parse().unwrap_or(usize::MAX), stem.to_string())
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn next_value(arguments: &mut impl Iterator<Item = String>, flag: &str) -> String {
    arguments
        .next()
        .unwrap_or_else(|| panic!("missing value after {flag}"))
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

fn path_string(path: &Path) -> &str {
    path.to_str().expect("benchmark paths must be valid UTF-8")
}

fn source_name(options: &Options) -> &'static str {
    if options.preload_memory {
        "preloaded-memory"
    } else {
        "parquet"
    }
}

fn string_type_name(options: &Options) -> &'static str {
    if options.utf8view { "Utf8View" } else { "Utf8" }
}

fn print_help() {
    println!(
        "Usage: cargo bench --bench workload -- [OPTIONS]\n\
         \n\
         --workload job|ceb-imdb|tpch Workload (default: job)\n\
         --data-dir PATH           Parquet root (defaults inside benchmark_data)\n\
         --query-dir PATH          SQL directory (defaults inside benchmark)\n\
         --queries NAMES           Comma-separated stems, e.g. 1a,6a (default: all)\n\
         --scale-factor N          TPC-H scale factor used in the default path\n\
         --threads N               DataFusion target partitions (default: 1)\n\
         --batch-size N            Override DataFusion's native Arrow batch size\n\
         --utf8view                Restore DataFusion 54.1's native Utf8View strings\n\
         --excitation-threshold F  Reactivate below this cardinality fraction (default: 1)\n\
         --warmups N               Untimed pairs per query (default: 1)\n\
         --runs N                  Timed pairs per query (default: 5)\n\
         --row-locations           Enable experimental cost-based row-location handoffs\n\
         --fresh-context-per-query Do not reuse Bloom samples across different queries\n\
         --parquet-pushdown        Enable Parquet late-materialized filter pushdown\n\
         --post-scan-membership    Evaluate Bloom membership after Parquet decoding\n\
         --predicate-cache-size N Override Parquet predicate cache bytes (0 disables)\n\
         --preload-memory          Diagnostic MemTable input (requires Bloom-only/audit/plan-only)\n\
         --reoptimize              Experimental: rerun P1 physical optimization after transfer\n\
         --log-transfer            Print Bloom transfer scheduling and phase timings\n\
         --show-metrics            Print executed-plan metrics after timing\n\
         --plan-only               Print baseline physical plans without executing queries\n\
         --handoff-audit           Execute Bloom only and report actual handoff counts\n\
         --baseline-only           Time stock DataFusion only with configured warmups/runs\n\
         --bloom-only              Time Bloom only with configured warmups/runs and medians\n\
         --show-plan               Print both physical plans after timing"
    );
}
