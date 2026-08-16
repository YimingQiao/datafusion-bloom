# Bloom materialization design

This document defines the boundary between Bloom propagation, DataFusion scan
execution, transfer handoffs, and formal join execution. The central rule is
that propagation scheduling and storage representation are independent.

## Responsibilities

Bloom makes four decisions at different layers:

1. **Propagation scheduling** selects the next source from cardinality,
   propagation lineage, and expected destination reduction. It does not inspect
   Parquet row locations, row-group density, payload width, or scan
   amplification.
2. **Predicate placement** decides whether membership is evaluated inside a
   data-source reader or after decoding. It may also derive sound storage
   pruning hints from an exact membership structure.
3. **Materialization** decides what data is retained when a table becomes a
   propagation source. This decision owns scan count, Arrow buffer ownership,
   and later in-memory compaction.
4. **Formal handoff** decides whether the stock DataFusion plan reads a
   `BloomCollection`, performs a direct filtered scan, or performs an
   experimental late materialization from retained row locations.

No storage-specific fact is allowed to affect step 1. Conversely, removing a
row-location guard from the scheduler does not remove the corresponding cost
check from steps 2–4.

## Correctness contract

For table operator `T`, a FullRows generation contains every P0 output row of
`T` that satisfies the local predicate and every transfer predicate committed
before that generation. It may contain extra rows because approximate
membership admits false positives; it must never omit a row that could
participate in the original join. Later generations are monotone subsets.

The handoff preserves the table operator's query-visible schema, null behavior,
and partitioned row multiset. Local predicates are not intentionally evaluated
twice, and filter-only reader columns are removed before publication. Direct
handoffs obey the same row condition without retaining a collection. The
formal join remains exact, so membership is only a safe semijoin reduction and
never substitutes for join equality. Sampling affects scheduling estimates
only and can never be used as an execution filter.

## Table lifecycle

Each query has two physical plans with the same native join tree. The formal
plan retains DataFusion's join-owned runtime dynamic filters. A separate P0
disables those runtime dependencies so its table operators can execute
independently during transfer. Each table starts with its P0 table operator, a
prepared reusable source sample, an estimated cardinality, and zero or more
pending transfer predicates.

```text
Unopened
  -> Sampled                   source sample is cached once per SessionContext
  -> Pending(filters...)       propagation installs membership
  -> SourceMaterialized        table becomes a propagation source
       -> FullRows             normal path, one table scan
       -> RowLocations         explicit experimental path
  -> Direct                    terminal destination, no retained source result
  -> FormalInput               stock DataFusion operators consume the handoff
```

On the normal path, becoming a propagation source creates `FullRows` exactly
once. Later incoming transfer predicates compact that collection in memory;
they do not trigger a Parquet rescan and do not convert the handoff to row
locations. A table that never becomes a source can remain `Direct` and perform
its only data scan during formal execution.

The prepared sample is setup state, not a per-query materialization. It is
built once for an immutable source and sample size, then reused across queries
in the same `SessionContext`. Query-specific local and transfer predicates are
still evaluated per query.

## DataFusion-specific mapping

The lifecycle follows DuckDB Bloom, but the physical containers and reader
interfaces are different:

| DuckDB Bloom concept | DataFusion Bloom equivalent | Consequence |
|---|---|---|
| `ColumnDataCollection` | partitioned Arrow `RecordBatch`es exposed through a custom `DataSource` | preserve partitions and expose exact handoff statistics |
| filter attached to the table scan | physical expression in `ParquetSource`'s row filter | membership can reject rows before payload decoding |
| storage min/max filtering | independent equality/range physical expressions | semantic membership and Parquet pruning remain separate |
| appending a filtered `DataChunk` flattens selected values | filtering an immutable Arrow array may retain shared page buffers | compact amplified arrays before retaining them |
| collection scan tasks | DataFusion execution partitions and async streams | preserve P0 file groups; never invent Bloom-only parallelism |
| query-owned collection | `Arc<BloomCollection>` plus `MemoryReservation` | ownership and release follow Rust lifetimes and DataFusion's pool |

There is no need to introduce a new join operator. The custom source is the
DataFusion analogue of a collection scan; formal execution continues to use
ordinary DataFusion joins, aggregates, exchanges, and output operators.

As in DuckDB Bloom, handoff replacement occurs after the engine has selected
its join plan. P0 exists only for transfer; formal execution starts from the
separately planned native DataFusion tree. Leaves with real handoffs are
replaced, while untouched scans retain their native dynamic filters. The
default therefore preserves DataFusion's join tree, build/probe sides, and
runtime behavior outside the handoff boundary. Running the physical optimizer
again is experimental: an exact handoff row count without matching NDV and
multiplicity statistics can underestimate a high-fanout intermediate and
reverse a good hash-join side.

## Default: one-pass FullRows

`HandoffPolicy::FullRows` is the default and matches Bloom's DuckDB lifecycle:

1. Reset the independently executable P0 table plan.
2. Attach all currently committed transfer predicates.
3. Push membership into the Parquet row filter when supported.
4. Attach exact integer equality or range expressions separately as Parquet
   pruning hints.
5. Execute local predicates and read every query-visible output column.
6. Coalesce small result batches and detach amplified Arrow backing buffers.
7. Retain the compact partitioned batches in `BloomCollection`.
8. Build outgoing membership from the retained exact rows.
9. Replace the corresponding leaf in the native formal plan with the
   collection.

The join phase remains DataFusion's normal `HashJoinExec` tree. FullRows means
that all columns exposed by the table operator are present in the handoff; it
does not mean that filter-only columns must escape the Parquet reader.

This path performs one Parquet data scan for a materialized source and no
second payload scan during formal execution.

## Predicate placement is not a handoff

The reader sees two logically different kinds of predicates:

- **membership** is the semantic transfer condition and must not produce false
  negatives;
- **pruning hints** are redundant, sound expressions used only to eliminate
  row groups or pages early.

For an exact singleton integer membership, Bloom submits both the membership
predicate and `column = value`. The equality often removes almost the entire
file through Parquet statistics/page indexes, while membership remains the
semantic authority. Wider exact integer bitmaps can provide min/max bounds;
approximate Bloom filters cannot provide exact pruning bounds.

Reader membership is ordered before local string predicates and payload
columns. Bloom knows its measured selectivity, while DataFusion's generic
reordering currently estimates only compressed bytes referenced by each
predicate. After filter pushdown, duplicate P0 predicates are removed and
projection pushdown is repeated.

`ParquetMembershipPlacement::PostScan` keeps membership above a scan boundary
for controlled comparison and unusual reader regressions. It does not alter
the scheduler or the FullRows/RowLocations choice.

## Arrow collection ownership

Parquet row filtering commonly returns tiny logical arrays that still retain
whole decoded page buffers. Keeping those arrays can turn 100 result rows into
tens of megabytes of reserved memory. `BloomCollection` therefore requires
collection-owned compact data:

- adjacent small batches are packed exactly up to DataFusion's configured
  batch size (except the final batch in each partition);
- ordinary arrays are copied only when physical memory is more than twice the
  logical result and at least 64 KiB would be released;
- every retained `Utf8View` and `BinaryView` uses Arrow view-buffer garbage
  collection, even when its total byte size is not otherwise amplified;
- compaction time is included in transfer and full query elapsed time;
- collection memory is reserved against DataFusion's memory pool after
  compaction.

Subsequent in-memory transfer filtering applies the same compaction rule.

## Resource and lifetime boundary

A handoff is query-scoped and is published to P1 only after its source scan
completes successfully. Cancellation or an error drops the partial handoff;
formal execution must never observe a partially built collection. The current
`BloomCollection` reserves its compact physical bytes in DataFusion's memory
pool and releases that reservation with the query-owned collection.

For inputs larger than the in-memory admission budget, the production extension
should add `SpilledFullRows`, not silently select row locations or rescan the
original Parquet source. It has the same logical contract as `FullRows`:

1. stream the one source scan through membership and local predicates;
2. compact batches, grow a memory reservation incrementally, and spill complete
   partitions to query-scoped Arrow IPC files when the budget is reached;
3. build outgoing membership while the same rows pass through the writer;
4. atomically publish an in-memory/on-disk collection after the scan succeeds;
5. let formal execution read that collection and delete spill files when the
   query finishes.

Admission and spilling belong to the materialization layer. Available memory,
estimated survivor bytes, and spill bandwidth must not change propagation
order. The current implementation is deliberately in-memory; its reservation
is made after collection compaction, so incremental reservation and spilling
remain required before treating unbounded-size FullRows as production-ready.

## Direct handoff

A destination with pending single-column membership that never becomes a
propagation source does not need a retained collection. Its original formal
scan receives membership, pruning hints, predicate ordering, and projection
cleanup in the same way as a FullRows scan. The exact join still verifies the
result. Direct therefore performs one scan and avoids unnecessary transfer
materialization.

Composite membership currently materializes as FullRows because it cannot be
represented by the same per-column direct-scan interface without losing tuple
correlation.

## Experimental RowLocations

`HandoffPolicy::CostBasedRowLocations` is explicitly opt-in. It may retain join
keys plus stable `(file_id, row_offset)` values and read query payload during
formal execution. It is useful only when all of the following hold:

- the table is wide and the retained transfer columns are much narrower;
- the expected survivor count is very small;
- selected positions are sufficiently concentrated;
- the extra Parquet scan and row-location generation cost less than retaining
  FullRows;
- the source is immutable, local, and has a stable canonical file layout.

If locality is scattered or the plan shape cannot preserve stable positions,
the policy returns FullRows. Once FullRows exists, it is never converted to
RowLocations. Row-location facts never enter propagation scheduling.

The current benchmark keeps this path experimental: on the synthetic wide
Parquet case, FullRows was faster than both one-pass and two-pass location
handoffs. Automatic two-pass FullRows was also rejected after JOB profiling;
generating positions for every fact row cost more than decoding a narrow
one-pass result on several queries.

## Scan-count invariants

| Lifecycle | Transfer data scans | Formal data scans | Normal total |
|---|---:|---:|---:|
| FullRows source | 1 | 0 | 1 |
| Direct destination | 0 | 1 | 1 |
| RowLocations source | 1 | 1 | 2 |
| Two-pass RowLocations source | 2 | 1 | 3 |

Prepared sampling is tracked separately and reused. Metadata/footer access is
also cached separately from data scans.

## Required profiling

Every materialization policy change is evaluated with complete query elapsed
time and must report at least:

- transfer initialization, collection, compaction, and formal execution time;
- scans per table and handoff kind;
- source/output rows and bytes;
- Parquet bytes scanned, row groups/pages pruned, and row-filter matches;
- predicate-cache inner records and final records;
- Arrow physical bytes before and after collection compaction;
- workload sum speedup, per-query speedup, and geometric mean.

JOB and TPC-H are run with the same DataFusion runtime, target partitions,
input files, query order, warmup policy, and prepared-sample lifetime for
Baseline and Bloom. The release comparison explicitly enables global Parquet
pushdown for both engines and uses the same ordinary-`Utf8` compatibility
setting. Native `Utf8View` and DataFusion-default Parquet settings remain
separate sensitivity runs so neither can become a hidden baseline difference.
