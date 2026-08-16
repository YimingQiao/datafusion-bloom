# Bloom for Apache DataFusion

[![Version: 0.1.0](https://img.shields.io/badge/version-0.1.0-blue)](Cargo.toml)
[![DataFusion: 54.1.0](https://img.shields.io/badge/DataFusion-54.1.0-orange)](Cargo.toml)
[![CI](https://github.com/YimingQiao/datafusion-bloom/actions/workflows/ci.yml/badge.svg)](https://github.com/YimingQiao/datafusion-bloom/actions/workflows/ci.yml)

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

DataFusion 54.1.0, one thread, one warmup plus three measured runs, planning
included, and complete result fingerprints checked against stock DataFusion:

| Workload | DataFusion | Bloom | Total speedup | Correct |
|---|---:|---:|---:|---:|
| JOB, 113 queries | 96.940 s | 65.291 s | **1.485×** | 113/113 |
| TPC-H SF10, 22 queries | 79.147 s | 68.728 s | **1.152×** | 22/22 |

Both sides use the same Parquet files, filter-pushdown settings, native batch
size, and ordinary Arrow `Utf8` mapping. The latter temporarily avoids a
DataFusion 54.1 `Utf8View` join-performance cliff; `--utf8view` restores the
native mapping for comparison.

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

## Running the benchmarks

The repository includes self-contained JOB and TPC-H runners. Prepare and run
them from the repository root:

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
- Related implementations: [Bloom for DuckDB](https://github.com/YimingQiao/bloom)
  and [BloomPG](https://github.com/YimingQiao/bloompg).

## License

Apache-2.0.
