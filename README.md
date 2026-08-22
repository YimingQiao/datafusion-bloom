# Bloom for Apache DataFusion

[![Version: 0.1.1](https://img.shields.io/badge/version-0.1.1-blue)](Cargo.toml)
[![DataFusion: 54.1.0](https://img.shields.io/badge/DataFusion-54.1.0-orange)](Cargo.toml)
[![CI](https://github.com/YimingQiao/datafusion-bloom/actions/workflows/ci.yml/badge.svg)](https://github.com/YimingQiao/datafusion-bloom/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Bloom gives Apache DataFusion a whole-query pre-join reduction stage. A
selective dimension can shrink fact tables several joins away before the first
formal join runs, while existing SQL and DataFusion's join engine stay
unchanged.

```text
SQL -> DataFusion optimized plan
             |
             v
  transfer: estimate -> propagate -> materialize
             |
             v
  join: the same DataFusion HashJoinExec tree
```

Bloom divides execution into two strict stages. **Transfer** propagates safe
join-key membership across the graph and produces compact table handoffs.
**Join** substitutes those handoffs into DataFusion's already optimized plan
and runs its stock joins, aggregates, exchanges, and output operators.

## What Bloom adds

- **Whole-graph transfer.** Membership can move in either direction and across
  multiple joins instead of stopping at one build-to-probe edge.
- **Adaptive execution.** Reusable samples suggest the first transfers; actual
  materialized cardinalities guide the next ones.
- **Reusable reduced inputs.** A participating table is normally scanned and
  materialized once. Later transfer predicates compact its Arrow handoff in
  memory rather than rescanning the source.
- **Early filtering.** Transfer membership runs before expensive local string
  predicates when the source supports it and can provide sound Parquet pruning
  bounds for exact integer domains.
- **Native formal execution.** Bloom does not add a join operator or rerun join
  ordering by default. Exact joins still determine the final result.
- **Fail-open planning.** Unsupported shapes keep DataFusion's native plan;
  recoverable transfer or memory-pool failures release temporary handoffs and
  use that same native plan.

## Algorithm lineage

Yannakakis' classic algorithm evaluates an acyclic join in two conceptual
steps: semijoin reduction removes tuples that cannot reach the answer, then the
reduced relations are joined. Predicate Transfer brought that pre-filtering
idea to general multi-join graphs with lightweight membership structures.
Robust Predicate Transfer (RPT) later studied how this family can remain robust
when the chosen join order is poor.

Bloom belongs to this research lineage, but it is its own design rather than a
rename or wrapper around RPT. Its defining boundary is an adaptive, executable
transfer stage followed by an unchanged engine-native join stage. This crate
implements that boundary with DataFusion physical plans, Parquet predicates,
and compact Arrow materializations.

## Results

DataFusion 54.1.0 on two Intel Xeon Platinum 8474C CPUs. Each query uses one
thread, the Parquet pages are prewarmed outside query timing, and the table
reports one complete measured pass over every query. SQL planning, transfer,
materialization, joins, aggregation, and full output consumption are included.

| Workload | Parquet | Queries | DataFusion | Bloom | Total speedup |
|---|---:|---:|---:|---:|---:|
| CEB IMDB | 1.43 GB, compressed | 3,133 | 3,948.177 s | 2,119.400 s | **1.863×** |
| JOB | 1.43 GB, compressed | 113 | 102.378 s | 69.398 s | **1.475×** |
| JOB | 2.58 GB, uncompressed | 113 | 78.025 s | 45.613 s | **1.711×** |
| STATS-CEB | 12.9 MB, compressed | 146 | 186.347 s | 174.083 s | **1.070×** |
| TPC-H SF10 | 2.47 GB, compressed | 22 | 83.506 s | 74.437 s | **1.122×** |

The default FullRows handoff also materializes DataFusion's physical scan
partitions in parallel. On the same machine and data, an eight-thread pass
gives:

| Workload | Parquet | Queries | DataFusion | Bloom | Total speedup |
|---|---:|---:|---:|---:|---:|
| CEB IMDB | 1.43 GB, compressed | 3,133 | 1,306.373 s | 533.061 s | **2.451×** |
| JOB | 1.43 GB, compressed | 113 | 28.065 s | 15.894 s | **1.766×** |
| JOB | 2.58 GB, uncompressed | 113 | 24.889 s | 11.253 s | **2.212×** |
| STATS-CEB | 12.9 MB, compressed | 146 | 182.136 s | 54.444 s | **3.345×** |
| TPC-H SF10 | 2.47 GB, compressed | 22 | 18.803 s | 13.636 s | **1.379×** |

All 3,527 Baseline/Bloom result pairs in each complete suite produced identical
complete-output fingerprints. Both sides use the same files, filter pushdown,
native 8,192-row batch size, and join dynamic filters. A prepared sample is
built once per immutable source and reused across its long-lived Bloom session.

The benchmark temporarily uses owned Arrow strings for both sides to avoid a
DataFusion 54.1 `Utf8View` join-performance cliff. CEB uses `LargeUtf8` because
some stock DataFusion intermediates exceed the 32-bit `Utf8` offset limit; the
other string workloads use `Utf8`. These are benchmark data representations,
not Bloom scheduling options. See the reproducibility notes below for the exact
commands and the guarded-cast normalization required by CEB.

## Compatibility

This preview supports the Cargo SemVer-compatible Apache DataFusion `54.x`
line starting at `54.1.0`, and Rust `1.88` or newer. CI tests both the committed
lockfile and a fresh latest-compatible dependency resolution. DataFusion major
versions may change physical-planner interfaces and receive a separately tested
Bloom release rather than being accepted silently.

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

## Running the benchmarks

The repository includes independently pinned preparation scripts for CEB IMDB,
JOB, STATS-CEB, and TPC-H, plus one runner for the full table. After preparing
the data as documented in [benchmark/README.md](benchmark/README.md), run all
workloads or a named subset from the repository root:

```bash
benchmark/scripts/run-benchmarks.sh
benchmark/scripts/run-benchmarks.sh ceb-imdb job-uncompressed stats-ceb
```

The runner includes planning, transfer, materialization, joins, and complete
output consumption in elapsed time, alternates Baseline and Bloom execution
order, and checks an order-independent fingerprint for every result. Raw logs
and environment metadata go to the ignored `benchmark_results/` directory.
Set `BLOOM_BENCH_WARMUPS=1 BLOOM_BENCH_RUNS=3` for per-query medians; both
Results tables use the script's practical full-suite default of one prewarmed
pass.
The parallel table uses
`BLOOM_BENCH_THREADS=8 benchmark/scripts/run-benchmarks.sh`.

## Configuration

Bloom works without query-specific tuning. `FullRows` is the normal handoff and
materializes every query-used column. Row locations are an opt-in experimental
late-materialization policy and never affect propagation scheduling:

```rust
let config = BloomConfig::default()
    .with_all_bounded_sources()
    .with_row_locations();
```

Prepared sampling is the default for a long-lived session. An ad-hoc service
can instead use query-local projected samples that are released after planning:

```rust
let config = BloomConfig::default()
    .with_all_bounded_sources()
    .with_instant_sampling();
```

The main configuration fields are:

| Field | Default | Purpose |
|---|---:|---|
| `enabled` | `true` | Enable or bypass Bloom planning |
| `sample_rows` | `10_000` | Target sample rows per immutable source |
| `sampling_mode` | `Prepared` | Reusable source samples or query-local `Instant` samples |
| `instant_parquet_row_groups` | `4` | Stratified candidate row groups in an Instant sample |
| `false_positive_rate` | `0.01` | Target for temporary probabilistic membership |
| `max_transfer_rounds` | `64` | Fixed-point safety bound |
| `excitation_threshold` | `1.0` | Cardinality fraction that reactivates a source |
| `handoff_policy` | `FullRows` | Formal transfer handoff representation |
| `parquet_membership_placement` | `Reader` | Reader-side or diagnostic post-scan membership |
| `reoptimize` | `false` | Experimental P1 physical reoptimization |

`with_transfer_logging()` prints propagation decisions, materialization phases,
handoff counts, and scan metrics. `with_post_scan_membership()` retains an A/B
path for investigating unusual Parquet-reader regressions.

FullRows handoffs and prepared samples reserve their retained Arrow bytes in
DataFusion's memory pool. If transfer cannot obtain or grow that reservation,
Bloom discards its temporary state and leaves the query on the native plan.
Prepared samples are single-flight across concurrent queries and retained in a
bounded session cache.

The workload runner accepts `--instant-sampling`. The one-command suite exposes
the same choice as `BLOOM_BENCH_SAMPLING=instant`; optionally set
`BLOOM_BENCH_INSTANT_ROW_GROUPS`. Its environment record and benchmark header
identify the selected mode.

## Correctness

The test suite includes randomized differential joins, nullable and composite
keys, multi-table propagation, Parquet reader placement, full-row and
row-location handoffs, projection remapping, aggregate partitioning, and
complete JOB/TPC-H output fingerprints. Approximate membership may retain extra
rows but cannot remove a row that could satisfy the original exact join.

## Related projects

Bloom, BloomPG, and Bloom for Apache DataFusion are sibling projects exploring
robust predicate transfer across different query engines.

- [Bloom](https://github.com/YimingQiao/bloom) — DuckDB extension.
- [BloomPG](https://github.com/YimingQiao/bloompg) — PostgreSQL extension.
- [Bloom for Apache DataFusion](https://github.com/YimingQiao/datafusion-bloom)
  — Apache DataFusion library (this repository).

## References

- Mihalis Yannakakis,
  [*Algorithms for Acyclic Database Schemes*](https://www.sigmod.org/publications/dblp/db/conf/vldb/Yannakakis81.html),
  VLDB 1981.
- Yifei Yang, Hangdong Zhao, Xiangyao Yu, and Paraschos Koutris,
  [*Predicate Transfer: Efficient Pre-Filtering on Multi-Join Queries*](https://arxiv.org/abs/2307.15255),
  CIDR 2024.
- Junyi Zhao, Kai Su, Yifei Yang, Xiangyao Yu, Paraschos Koutris, and Huanchen
  Zhang,
  [*Debunking the Myth of Join Ordering: Toward Robust SQL Analytics*](https://arxiv.org/abs/2502.15181),
  SIGMOD 2025.
- Yiming Qiao, Peter Boncz, and Huanchen Zhang,
  [*Robust Predicate Transfer with Dynamic Execution*](https://duckdb.org/library/robust-predicate-transfer-vldb/),
  PVLDB 2026.

## License

MIT.
