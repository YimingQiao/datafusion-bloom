# Bloom for Apache DataFusion

[![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-blue)](Cargo.toml)
[![DataFusion: 54.1.0](https://img.shields.io/badge/DataFusion-54.1.0-orange)](Cargo.toml)
[![CI](https://github.com/YimingQiao/datafusion-bloom/actions/workflows/ci.yml/badge.svg)](https://github.com/YimingQiao/datafusion-bloom/actions/workflows/ci.yml)

Bloom speeds up complex join queries in Apache DataFusion by moving selective
membership across the join graph before the joins execute. It divides query
execution into two stages:

1. **transfer** executes independent table operators, propagates join-key
   membership, and produces reduced handoffs;
2. **join** runs the original query with DataFusion's stock physical operators.

Bloom does not replace `HashJoinExec` or implement a second join engine. The
formal plan remains exact; transfer only removes rows that cannot contribute to
the result.

## How it works

- **Estimate.** Reusable prepared samples estimate locally filtered table
  cardinalities and promising propagation edges.
- **Transfer and adapt.** Bloom chooses a source, moves membership to its
  neighbors, observes the resulting cardinalities, and chooses again.
- **Materialize once.** A participating source normally scans once and retains
  all query-used columns in a compact Arrow handoff. Later transfer predicates
  filter that handoff in memory instead of rescanning the source.
- **Filter early.** Transfer membership is evaluated before expensive local
  string predicates when the source supports it. Exact integer membership may
  also provide independent row-group or page-pruning bounds.
- **Join normally.** The reduced handoffs are substituted into DataFusion's
  already optimized physical plan. Its join tree, build/probe choices, and
  stock execution operators are preserved.

On DataFusion, a partitioned collection of immutable Arrow `RecordBatch`es is
the equivalent of Bloom's materialized table collection. The collection
publishes exact statistics for diagnostics, is charged to DataFusion's memory
pool, and keeps the partitioning chosen for the original table plan. Bloom
does not rerun join ordering by default: row counts alone do not describe
duplicate-heavy join-key distributions.

## Compatibility

This preview targets exactly Apache DataFusion `54.1.0` and Rust `1.88` or
newer. The exact dependency is intentional because the implementation uses
DataFusion physical-plan, filter-pushdown, and Parquet-reader interfaces that
can change between releases.

Bloom currently supports bounded inner equi-join graphs whose join keys can be
traced to table columns, including nullable and composite keys. Unsupported
outer joins, untraceable expressions, and unsafe dependent subplans fall back
to the original DataFusion plan.

## Build and test

```bash
cargo build --release
cargo test --lib --tests
cargo clippy --lib --tests --benches -- -D warnings
```

## Use

Install Bloom's query planner into an existing `SessionState`:

```rust
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_bloom::{BloomConfig, install_bloom};

let state = SessionStateBuilder::new_with_default_features().build();
let state = install_bloom(state, BloomConfig::default())?;
let context = SessionContext::new_with_state(state);
```

The conservative default accepts repeatable in-memory sources. Enable all
bounded sources, including immutable Parquet tables, explicitly:

```rust
let config = BloomConfig::default().with_all_bounded_sources();
let state = install_bloom(state, config)?;
```

## Benchmarks

The repository includes self-contained runners for the complete 113-query Join
Order Benchmark and all 22 TPC-H queries. Prepare and run them from the
repository root:

```bash
benchmark/scripts/prepare-job.sh
cargo bench --bench workload -- \
  --workload job --threads 1 --warmups 1 --runs 3 --parquet-pushdown

benchmark/scripts/prepare-tpch.sh 10
cargo bench --bench workload -- \
  --workload tpch --scale-factor 10 --threads 1 --warmups 1 --runs 3 \
  --parquet-pushdown
```

The runner includes planning, transfer, materialization, joins, and complete
output consumption in elapsed time, alternates Baseline and Bloom execution
order, and checks an order-independent fingerprint for every result. See
[benchmark/README.md](benchmark/README.md) for data provenance and command-line
options.

## Configuration

Bloom works without query-specific tuning. `FullRows` is the normal handoff and
materializes every query-used column. Row locations are an opt-in experimental
late-materialization policy and never affect propagation scheduling:

```rust
let config = BloomConfig::default()
    .with_all_bounded_sources()
    .with_row_locations();
```

The main configuration fields are:

| Field | Default | Purpose |
|---|---:|---|
| `enabled` | `true` | Enable or bypass Bloom planning |
| `sample_rows` | `10_000` | Prepared sample rows per immutable source |
| `false_positive_rate` | `0.01` | Target for temporary probabilistic membership |
| `max_transfer_rounds` | `64` | Fixed-point safety bound |
| `excitation_threshold` | `1.0` | Cardinality fraction that reactivates a source |
| `handoff_policy` | `FullRows` | Formal transfer handoff representation |
| `parquet_membership_placement` | `Reader` | Reader-side or diagnostic post-scan membership |
| `reoptimize` | `false` | Experimental P1 physical reoptimization |

`with_transfer_logging()` prints propagation decisions, materialization phases,
handoff counts, and scan metrics. `with_post_scan_membership()` retains an A/B
path for investigating unusual Parquet-reader regressions.

## Correctness

The test suite includes randomized differential joins, nullable and composite
keys, multi-table propagation, Parquet reader placement, full-row and
row-location handoffs, projection remapping, aggregate partitioning, and
complete JOB/TPC-H output fingerprints. Approximate membership may retain extra
rows but cannot remove a row that could satisfy the original exact join.

## License

Apache-2.0.
