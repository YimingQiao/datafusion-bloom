# Materialization redesign and benchmark results — 2026-08-16

> Historical design result, superseded for performance claims by
> [the 2026-08-17 release benchmark](RESULTS-2026-08-17.md). Its 16-thread
> tables predate the common-`Utf8`, strong-baseline, native-formal-plan
> protocol.

This report records the implementation and validation of Bloom's DataFusion
materialization lifecycle. It follows the earlier
[benchmark fairness audit](RESULTS-2026-08-16.md), which remains the source for
the runner's concurrency correction but no longer describes the current
implementation or performance.

## Decision

Bloom now uses a one-pass, compact `FullRows` handoff by default:

1. propagation scheduling considers cardinality, lineage, propagation edges,
   and expected reduction only;
2. when a table becomes a source, its independently executable P0 table plan is
   scanned once with every committed transfer predicate and every query-visible
   output column;
3. Parquet evaluates transfer membership in its row filter by default, while
   exact integer equality/range expressions are supplied independently for
   statistics and page pruning;
4. selected Arrow data is detached from amplified reader buffers and retained
   in a query-owned `BloomCollection`;
5. later transfer predicates compact the collection in memory instead of
   rescanning Parquet;
6. P1 reads the collection and executes the original DataFusion join tree.

A terminal single-column destination may use `Direct`: it is not retained in
transfer and receives membership in its only formal scan. `RowLocations` is an
explicit experimental handoff and is disabled by default. Predicate placement,
handoff representation, and propagation scheduling are independent decisions.

The complete rationale, correctness contract, scan-count invariants, and
resource boundary are in [the materialization design](../docs/MATERIALIZATION.md).

## Why DataFusion needed different mechanics

DuckDB Bloom can hand a compact `ColumnDataCollection` directly to its
collection scan. DataFusion's equivalent is a partitioned set of immutable
Arrow `RecordBatch`es exposed as a custom `DataSource`. That difference caused
three important implementation issues:

- filtering an Arrow view can retain the complete decoded Parquet page buffer;
- reader membership and Parquet row-group/page pruning are separate mechanisms;
- replacing DataFusion's file groups can accidentally give Bloom more scan
  parallelism than Baseline.

The redesign therefore preserves P0 file groups, reserves compact collection
bytes in DataFusion's memory pool, garbage-collects Arrow view buffers, copies
only materially amplified ordinary arrays, and publishes exact collection
statistics to P1. It does not add a join operator.

## Stable complete-workload results

Every row below uses 16 Tokio workers and DataFusion target partitions, one
warmup, and three measured Baseline/Bloom pairs per query. Times are per-query
medians summed over the complete workload. The timer begins before SQL
planning and ends after the complete Arrow output is materialized. Every output
fingerprint matched; a mismatch would abort the runner. Prepared immutable
source samples are created once per long-lived `SessionContext` and reused.

`DataFusion default` leaves DataFusion 54.1's general Parquet row-filter
pushdown option at its default. Bloom still places its own transfer membership
inside the reader. `Both push down local filters` explicitly enables the same
general Parquet option for Baseline and Bloom; on JOB this is the stronger
control because it greatly improves Baseline independently of Bloom.

| Workload | Scan configuration | DataFusion | Bloom | Workload speedup | Query geomean | Faster queries |
|---|---|---:|---:|---:|---:|---:|
| JOB, 113 queries | DataFusion default | 37,741.623 ms | 14,304.084 ms | **2.639x** | **1.556x** | 92/113 |
| JOB, 113 queries | Both push down local filters | 21,072.732 ms | 13,727.180 ms | **1.535x** | **1.462x** | 94/113 |
| TPC-H-derived SF10, 22 queries | DataFusion default | 7,991.222 ms | 7,595.301 ms | **1.052x** | **1.023x** | 8/22 |
| TPC-H-derived SF10, 22 queries | Both push down local filters | 10,993.981 ms | 8,375.786 ms | **1.313x** | **1.210x** | 16/22 |

The second TPC-H row is a sensitivity result, not a preferred headline:
globally enabling reader filters made its DataFusion baseline slower on this
workload. The default TPC-H result is the representative result and is close to
neutral overall. Q7, Q8, Q17, Q19, and Q21 benefit; Q2, Q3, Q9, Q10, and Q16
still spend more in transfer than they save.

The controlled JOB row is the strongest evidence that the result is not merely
caused by a disabled Baseline scan option. Even after Baseline drops from 37.7
seconds to 21.1 seconds, Bloom is faster on 94 of 113 queries and retains a
1.462x per-query geometric mean.

## Materialization profiling evidence

### Reader placement

JOB 3a was run at one thread with one warmup, three measured pairs, and general
Parquet pushdown enabled for both sides:

| Bloom membership placement | Bloom elapsed | Speedup over same-run Baseline |
|---|---:|---:|
| reader-side (default) | 936.510 ms | 1.269x |
| post-scan diagnostic | 1,195.741 ms | 0.978x |

Reader placement is 1.277x faster than post-scan placement for this case. The
remaining bottleneck is nevertheless in DataFusion's Parquet reader: the
`movie_info` source produces only 1,533 rows but reads about 239 MB and takes
about 953 ms in a cold one-thread profile. Sparse key membership combined with
a string local predicate creates substantial decode/predicate-cache
amplification. This is now isolated as a scan-placement/reader problem; changing
the handoff lifecycle or propagation order is not an appropriate workaround.

### Arrow ownership

The same current JOB 3a profile showed why logical row counts are not enough
for a retained Arrow collection:

| Source | Retained rows | Reader-backed bytes | Compact bytes | Compact time |
|---|---:|---:|---:|---:|
| `keyword` | 30 | 262,240 | 216 | 0.012 ms |
| `movie_keyword` | 12,951 | 5,298,600 | 103,800 | 0.032 ms |
| `movie_info` | 1,533 | 8,580 | 6,228 | 0.022 ms |
| `title` | 105 | 29,848,208 | 4,419 | 0.042 ms |

The 105-row `title` result retained 28.5 MB before Arrow view-buffer garbage
collection. Keeping it unmodified would charge and carry those Parquet buffers
through the formal join despite the tiny logical result.

### Membership versus pruning

JOB 32a propagates a singleton integer key. Membership alone previously probed
the full `movie_keyword` input. The independent equality hint now reduces its
20 row groups to 4, prunes 14 of 19 indexed pages, reads about 797 KB, and
collects the one surviving row in about 3.2 ms. The query still regresses
because its native execution is only about 15 ms and fixed transfer planning
does not amortize; the storage scan itself is no longer the dominant mistake.

## Rejected alternatives

Automatic two-pass FullRows was prototyped and removed. On ten representative
JOB regressions at one thread, blanket two-pass materialization reduced the
query geomean from 1.151x to 1.043x; a width-gated version reached only 1.118x.
For example, discovering narrow keys still scanned the full source, while
sparse positions touched most Parquet pages again when payload was read.

The explicit synthetic wide-table handoff benchmark also favored FullRows:

| Path | Complete query elapsed |
|---|---:|
| DataFusion | 119.131 ms |
| Bloom FullRows | 131.158 ms |
| Bloom RowLocations | 168.624 ms |

RowLocations was 1.286x slower than FullRows even in this deliberately wide
case. It therefore remains opt-in behind a conservative width, selectivity,
payload, and locality policy. No row-location fact enters the propagation
scheduler, and an existing FullRows collection is never converted to
locations.

## Remaining system work

The current in-memory collection is appropriate for these benchmarks and is
accounted in DataFusion's memory pool, but it reserves memory after collection
and compaction. A production large-input implementation needs an incrementally
reserved, spillable FullRows collection. Spill must retain FullRows semantics,
build outgoing membership in the same source pass, and make formal execution
read the spill rather than rescan the original Parquet table.

The next performance work is deliberately narrower than the materialization
redesign:

1. add a transfer activation profitability guard for very short queries such
   as JOB 32a/32b without using storage facts in propagation ordering;
2. profile TPC-H Q2/Q3/Q9/Q10/Q16 and avoid transfers whose complete cost does
   not amortize;
3. investigate DataFusion's sparse Parquet row-filter predicate cache and page
   decode envelope, retaining reader/post-scan placement as an independent A/B
   switch;
4. implement incremental reservation and spill before claiming production
   support for arbitrarily large FullRows handoffs.

## Reproduction

```bash
# Representative DataFusion-default results
cargo bench --bench workload -- \
  --workload job --threads 16 --warmups 1 --runs 3
cargo bench --bench workload -- \
  --workload tpch --scale-factor 10 --threads 16 --warmups 1 --runs 3

# Control: enable general Parquet local-filter pushdown for both engines
cargo bench --bench workload -- \
  --workload job --threads 16 --warmups 1 --runs 3 --parquet-pushdown

# Reader/post-scan membership A/B
cargo bench --bench workload -- \
  --workload job --queries 3a --threads 1 --warmups 1 --runs 3 \
  --parquet-pushdown
cargo bench --bench workload -- \
  --workload job --queries 3a --threads 1 --warmups 1 --runs 3 \
  --parquet-pushdown --post-scan-membership
```
