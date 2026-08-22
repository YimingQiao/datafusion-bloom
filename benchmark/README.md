# Bloom benchmark suite

This directory owns the reproducible end-to-end benchmark for this project.
It never reads data, queries, binaries, or caches from another local project.

The release benchmark uses four standard workload families:

- JOB: the 113 canonical Join Order Benchmark queries and the May 2013 IMDB
  research snapshot, stored as both compressed and uncompressed Parquet.
- CEB IMDB: 3,133 cardinality-estimation benchmark queries over the same IMDB
  snapshot.
- STATS-CEB: 146 multi-join queries over simplified Stack Overflow data.
- TPC-H: the 22 standard decision-support queries over independently generated
  scale-factor data.

Generated data lives below `benchmark_data/` and measurements below
`benchmark_results/`; both directories are intentionally ignored by Git.
Preparing data is an offline step and is never included in query timing.

The workload runner measures wall-clock time from SQL parsing through complete
Arrow result materialization. Planning, Bloom transfer, handoff creation,
joins, aggregation, and output are therefore all included. Baseline and Bloom
runs alternate, use identical DataFusion settings, and compare deterministic
order-independent result fingerprints before reporting a speedup. `--threads`
sets both DataFusion `target_partitions` and the exact Tokio worker count; one
thread uses a current-thread runtime. Bloom preserves P0 scan file groups, so a
handoff cannot acquire hidden scan parallelism.

The runner does not override DataFusion's 8,192-row Arrow batch size by default.
`--batch-size` exists only for sensitivity
analysis. Baseline and Bloom use the same Parquet tables and retain
DataFusion's native join dynamic-filter pushdown.

The runner temporarily maps strings to owned Arrow offset arrays for both
engines because DataFusion's native `Utf8View` `take` path can retain an
amplified backing-buffer graph across high-fanout joins. Ordinary `Utf8` is the
default. Full CEB IMDB uses `LargeUtf8`, described below, because some stock
DataFusion intermediates exceed the 32-bit offset limit. The result header
always prints the effective representation. `--utf8view` restores the native
default for a sensitivity run. These common settings do not change Bloom
scheduling or handoff policy.

`--preload-memory` is a materialization diagnostic, not a speedup
configuration: `MemTable` does not expose the same dynamic-filter path as
Parquet, so its Baseline/Bloom ratio is not representative.

The default `shared-workload` mode uses one long-lived pair of sessions. A
Bloom prepared sample is built once per immutable source and sampling size and
then reused, as it would be by a persistent service. Use
`--fresh-context-per-query --warmups 0 --runs 1` only to diagnose cold
single-query latency without cross-query prepared-sample reuse; context and
table registration remain outside the query timer.

## Reproducing the complete suite

Prepare each independently owned data cache from the repository root:

```bash
benchmark/scripts/prepare-job.sh
BLOOM_JOB_PARQUET_DIR="$PWD/benchmark_data/job/parquet-uncompressed" \
BLOOM_JOB_COMPRESSION=uncompressed \
  benchmark/scripts/prepare-job.sh
BLOOM_JOB_PARQUET_DIR="$PWD/benchmark_data/job/parquet-largeutf8" \
BLOOM_JOB_STRING_TYPE=large-utf8 \
  benchmark/scripts/prepare-job.sh
benchmark/scripts/prepare-ceb-imdb.sh
benchmark/scripts/prepare-stats-ceb.sh
benchmark/scripts/prepare-tpch.sh 10
```

Then run every workload, or name a subset. The script prewarms the selected
Parquet files outside query timing, records the environment, uses one complete
measured pass by default, and writes raw logs only to the ignored
`benchmark_results/` directory:

```bash
benchmark/scripts/run-benchmarks.sh
benchmark/scripts/run-benchmarks.sh ceb-imdb stats-ceb
```

For per-query medians instead of the practical full-suite pass:

```bash
BLOOM_BENCH_WARMUPS=1 BLOOM_BENCH_RUNS=3 \
  benchmark/scripts/run-benchmarks.sh job-compressed tpch
```

No run-script path changes DataFusion's batch size. Set
`BLOOM_BENCH_THREADS` to change both the Tokio worker count and DataFusion
target partitions; the README reports complete one- and eight-thread tables.
Instant runs accept `BLOOM_BENCH_INSTANT_ROW_GROUPS` for sampling sensitivity
checks. `BLOOM_BENCH_PREDICATE_CACHE_SIZE` defaults to zero for both engines to
avoid the Arrow 59.2
[sparse-page predicate-cache regression](https://github.com/apache/arrow-rs/issues/10733);
remove this common workaround after the upstream fix is released.

## CEB IMDB provenance

`benchmark/scripts/prepare-ceb-imdb.sh` downloads and checksum-verifies the
3,133-query corpus at commit `1f39e9aa85ee64249f60bfa59543e8707b228644`
of <https://github.com/RyanMarcus/imdb_pg_dataset>. It uses the independently
prepared compressed JOB Parquet data and recursively preserves the original
query groups.

The full-suite CEB run reads the same values from the separate
`parquet-largeutf8` directory. A small number of high-fanout stock DataFusion
plans materialize more than 2 GB of selected string payload and overflow
Arrow's 32-bit `Utf8` offsets; native `Utf8View` avoids that limit but exhibits
the buffer-retention cliff described above. `LargeUtf8` keeps owned,
copy-on-selection strings while changing only the offset width. The
`--large-utf8` runner flag verifies that every string column actually uses this
representation; Baseline and Bloom always share the same files.

The source corpus also contains 435 queries whose numeric text predicates use
`regex AND value::float`. DataFusion can reorder those conjuncts and evaluate
the fallible cast on a nonnumeric row before the regular expression. The runner
leaves the pinned SQL files untouched and normalizes only the two guarded CEB
casts to `TRY_CAST(value AS FLOAT)` at load time. This preserves the workload's
intended false-for-nonnumeric-row behavior for both Baseline and Bloom and is
reported here rather than silently dropping the affected queries.

## JOB provenance

The standard IMDB snapshot is downloaded directly from the CWI JOB research
archive: <https://event.cwi.nl/da/job/>. The archive's notice limits its use to
database-engineering and scientific-research experiments. The canonical query
text is pinned to commit `a39603662e023e449cb2121997a5034df9e02ebf` of
<https://github.com/gregrahn/join-order-benchmark>.

Prepare and run JOB from the repository root:

```bash
benchmark/scripts/prepare-job.sh
cargo bench --bench workload -- \
  --workload job --threads 1 --warmups 1 --runs 3 --parquet-pushdown \
  --predicate-cache-size 0
```

Use `--queries 1a,6a,13a` for a smoke subset. Add `--show-plan` to print the
physical plans after the timed runs.

The default JOB files use `zstd(3)`. A separate uncompressed copy can be
prepared without replacing it, which is useful when comparing with the
uncompressed JOB result reported by the DuckDB Bloom project:

```bash
BLOOM_JOB_PARQUET_DIR="$PWD/benchmark_data/job/parquet-uncompressed" \
BLOOM_JOB_COMPRESSION=uncompressed \
  benchmark/scripts/prepare-job.sh
cargo bench --bench workload -- \
  --workload job \
  --data-dir benchmark_data/job/parquet-uncompressed \
  --threads 1 --warmups 1 --runs 3 --parquet-pushdown \
  --predicate-cache-size 0 --bloom-only
```

To compare against the Bloom README protocol without also paying for the
DataFusion baseline, use the same-process Bloom-only mode. It still measures
complete query elapsed time and fingerprints the complete output:

```bash
cargo bench --bench workload -- \
  --workload job --threads 1 --warmups 1 --runs 3 \
  --parquet-pushdown --predicate-cache-size 0 --bloom-only
```

The default benchmark uses `FullRows`. The experimental cost-based
row-location handoff is enabled separately with `--row-locations`; both modes
honor the same DataFusion scan partitioning:

```bash
cargo bench --bench workload -- \
  --workload job --threads 1 --warmups 0 --runs 1 \
  --row-locations
```

## TPC-H

TPC-H setup and its pinned generator/query provenance are maintained by
`benchmark/scripts/prepare-tpch.sh`. Data is generated by the Apache-2.0
`tpchgen-cli` 3.0.0 crate; the SQLBench-H text is pinned to Apache
DataFusion Benchmarks commit `cb12c981e6608e0f2dcf919956ada8f1f1622d72`
and kept in this repository. For example:

```bash
benchmark/scripts/prepare-tpch.sh 10
cargo bench --bench workload -- \
  --workload tpch --scale-factor 10 --threads 1 --warmups 1 --runs 3 \
  --parquet-pushdown --predicate-cache-size 0
```

## STATS-CEB provenance

`benchmark/scripts/prepare-stats-ceb.sh` pins commit
`670cb8d4bf4cbfa32f94fdf17f33973d3fd67d1b` of
<https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark>, verifies its
archive checksum, extracts exactly the eight simplified Stack Overflow CSV
tables and 146 workload queries, and converts the tables to Zstandard Parquet
with the repository's Rust preparation program:

```bash
benchmark/scripts/prepare-stats-ceb.sh
cargo bench --bench workload -- \
  --workload stats-ceb --threads 1 --warmups 1 --runs 3 \
  --parquet-pushdown --predicate-cache-size 0
```

## Diagnostics

`--log-transfer` prints table estimates, excitation rounds, policy decisions,
handoff modes, scan metrics, and transfer/finalization time. The result table
reports separate `full_rows`, `row_locations`, and `direct` counts.
`--post-scan-membership` moves Bloom membership above the Parquet reader for an
A/B diagnosis; reader placement is the default. `--parquet-pushdown` enables
DataFusion's general Parquet filter pushdown for both Baseline and Bloom.
`--handoff-audit` executes Bloom only, fully materializes and fingerprints each
output, and reports those counts without paying for a baseline run; it is the
fast way to audit a policy over every workload query.
`--bloom-only` is the measured counterpart: it honors `--warmups` and `--runs`
and reports per-query medians plus their workload sum.
`--preload-memory` loads the Parquet tables into shared Arrow `MemTable`s before
timing and reports that setup separately. It is useful for isolating transfer
CPU and handoff costs, but must not be used to claim a DataFusion speedup because
the source capabilities and resulting physical plan differ from Parquet.
`--show-metrics` prints
the executed baseline and Bloom physical-plan metrics after timing; formatting
those metrics is deliberately outside the measured interval. `--show-plan`
prints fresh physical plans, and `--plan-only` performs no formal query
execution. `--reoptimize` is an experimental diagnostic that reruns P1 after
transfer; release results preserve DataFusion's original join tree and sides.

Every timed run materializes the complete Arrow output and verifies an
order-independent fingerprint containing row count, two independent sums, and
an XOR component. A mismatch aborts the workload immediately.

`cargo bench --bench handoff` compares FullRows and RowLocations on a generated
high-entropy wide Parquet table. Data generation and warmup occur outside its
measured complete-query runs.
