//! Streaming collection and memory accounting for the default FullRows handoff.

use super::*;

const MIN_COMPACTION_SCRATCH_BYTES: usize = 64 * 1024;

/// Shared accounting for batches that have crossed Bloom's ownership
/// boundary. The initial estimate remains an admission check; retained and
/// transient bytes are verified incrementally, then become the final charge.
struct StreamingHandoffReservation {
    state: Mutex<StreamingHandoffReservationState>,
}

struct StreamingHandoffReservationState {
    reservation: MemoryReservation,
    retained_bytes: usize,
    transient_bytes: usize,
    peak_reserved_bytes: usize,
}

impl StreamingHandoffReservation {
    fn new(reservation: MemoryReservation, estimated_bytes: usize) -> Result<Arc<Self>> {
        reservation.try_grow(estimated_bytes)?;
        let peak_reserved_bytes = reservation.size();
        Ok(Arc::new(Self {
            state: Mutex::new(StreamingHandoffReservationState {
                reservation,
                retained_bytes: 0,
                transient_bytes: 0,
                peak_reserved_bytes,
            }),
        }))
    }

    /// Add a reader batch to the transient side of the ownership boundary.
    /// Buffered partial batches from every partition remain accounted here.
    fn begin_input(&self, bytes: usize, compaction_input_bytes: usize) -> Result<()> {
        let mut state = self.lock()?;
        let transient_bytes = state.transient_bytes.saturating_add(bytes);
        // Compaction can briefly own both the reader group and a fresh Arrow
        // allocation. The fixed slack covers allocator/array metadata when a
        // canonical output is slightly larger than its logical input.
        let scratch_bytes = compaction_input_bytes.saturating_add(MIN_COMPACTION_SCRATCH_BYTES);
        let required = state
            .retained_bytes
            .saturating_add(transient_bytes)
            .saturating_add(scratch_bytes);
        if required > state.reservation.size() {
            state
                .reservation
                .try_grow(required - state.reservation.size())?;
        }
        state.transient_bytes = transient_bytes;
        state.peak_reserved_bytes = state.peak_reserved_bytes.max(state.reservation.size());
        Ok(())
    }

    /// Atomically replace consumed reader buffers with compacted output and
    /// the partial reader-owned group that remains buffered for this stream.
    fn commit_compaction(
        &self,
        consumed_transient_bytes: usize,
        retained_bytes: usize,
        buffered_transient_bytes: usize,
    ) -> Result<()> {
        let mut state = self.lock()?;
        let transient_bytes = state
            .transient_bytes
            .checked_sub(consumed_transient_bytes)
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "Bloom streaming transient memory accounting underflow".to_string(),
                )
            })?
            .saturating_add(buffered_transient_bytes);
        let retained_bytes = state.retained_bytes.saturating_add(retained_bytes);
        let required = retained_bytes.saturating_add(transient_bytes);
        if required > state.reservation.size() {
            state
                .reservation
                .try_grow(required - state.reservation.size())?;
        }
        state.retained_bytes = retained_bytes;
        state.transient_bytes = transient_bytes;
        state.peak_reserved_bytes = state.peak_reserved_bytes.max(state.reservation.size());
        Ok(())
    }

    /// Convert the shared streaming account into the reservation owned by the
    /// completed FullRows handoff and release any unused estimate or scratch.
    fn finish(self: Arc<Self>) -> Result<(MemoryReservation, usize, usize)> {
        let tracker = Arc::try_unwrap(self).map_err(|_| {
            DataFusionError::Internal(
                "Bloom streaming reservation still had active collectors".to_string(),
            )
        })?;
        let state = tracker.state.into_inner().map_err(|_| {
            DataFusionError::Internal("Bloom streaming reservation lock was poisoned".to_string())
        })?;
        if state.transient_bytes != 0 {
            return Err(DataFusionError::Internal(format!(
                "Bloom streaming collection retained {} bytes of reader state",
                state.transient_bytes
            )));
        }
        state.reservation.try_resize(state.retained_bytes)?;
        Ok((
            state.reservation,
            state.retained_bytes,
            state.peak_reserved_bytes,
        ))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, StreamingHandoffReservationState>> {
        self.state.lock().map_err(|_| {
            DataFusionError::Internal("Bloom streaming reservation lock was poisoned".to_string())
        })
    }
}

struct StreamedFullRowsPartition {
    batches: Vec<RecordBatch>,
    reader_bytes: usize,
    input_batches: usize,
    compact_elapsed: Duration,
}

/// Consume one physical scan partition without first retaining all of its
/// reader output. A single partial target batch is the only reader-owned state
/// allowed to survive across stream polls.
async fn collect_full_rows_partition(
    mut stream: SendableRecordBatchStream,
    target_batch_rows: usize,
    reservation: Arc<StreamingHandoffReservation>,
) -> Result<StreamedFullRowsPartition> {
    let mut builder = MaterializedPartitionBuilder::new(target_batch_rows);
    let mut batches = Vec::new();
    let mut reader_bytes = 0_usize;
    let mut input_batches = 0_usize;
    let mut compact_elapsed = Duration::ZERO;
    let mut buffered_bytes = 0_usize;

    while let Some(batch) = stream.try_next().await? {
        let batch_bytes = batch_physical_bytes(&batch);
        reader_bytes = reader_bytes.saturating_add(batch_bytes);
        input_batches += 1;

        reservation.begin_input(batch_bytes, buffered_bytes.saturating_add(batch_bytes))?;
        let compact_started = Instant::now();
        let compacted = builder.push(batch)?;
        compact_elapsed += compact_started.elapsed();
        let compacted_bytes = compacted.iter().map(batch_physical_bytes).sum::<usize>();
        let next_buffered_bytes = builder.buffered_physical_bytes();
        reservation.commit_compaction(
            buffered_bytes.saturating_add(batch_bytes),
            compacted_bytes,
            next_buffered_bytes,
        )?;
        buffered_bytes = next_buffered_bytes;
        batches.extend(compacted);
    }

    let compact_started = Instant::now();
    let final_batch = builder.finish()?;
    compact_elapsed += compact_started.elapsed();
    let final_bytes = final_batch
        .as_ref()
        .map(batch_physical_bytes)
        .unwrap_or_default();
    reservation.commit_compaction(buffered_bytes, final_bytes, 0)?;
    if let Some(batch) = final_batch {
        batches.push(batch);
    }

    Ok(StreamedFullRowsPartition {
        batches,
        reader_bytes,
        input_batches,
        compact_elapsed,
    })
}

/// Build Bloom's default handoff by reading every query-required column,
/// applying current transfer restrictions, and resetting Arrow ownership
/// before the batches enter formal execution.
pub(super) async fn collect_full_rows_handoff(
    request: FullRowsRequest<'_>,
    context: Arc<TaskContext>,
) -> Result<TransferHandoff> {
    let FullRowsRequest {
        table,
        filters,
        input_row_hint,
        expected_rows,
        full_row_width,
        parquet_membership_placement,
        log_transfer_steps,
    } = request;
    let prepare_started = Instant::now();
    // Preserve P0's file groups and partition count. DataFusion has already
    // sized them from `target_partitions`; creating one transfer partition per
    // physical file here would let Bloom exceed the caller's concurrency while
    // the baseline remains constrained to the configured target.
    let plan = transfer_scan_plan(
        Arc::clone(&table.plan),
        filters,
        context.as_ref(),
        parquet_membership_placement,
    )?;
    let reservation = MemoryConsumer::new("BloomTransferHandoff").register(context.memory_pool());
    let reservation = StreamingHandoffReservation::new(
        reservation,
        expected_rows.saturating_mul(full_row_width),
    )?;
    let prepare_elapsed = prepare_started.elapsed();
    let collect_started = Instant::now();
    let streams = execute_stream_partitioned(Arc::clone(&plan), Arc::clone(&context))?;
    let target_batch_rows = context.session_config().batch_size();
    let streamed = try_join_all(streams.into_iter().map(|stream| {
        collect_full_rows_partition(stream, target_batch_rows, Arc::clone(&reservation))
    }))
    .await?;
    let collect_elapsed = collect_started.elapsed();
    let reader_bytes = streamed
        .iter()
        .map(|partition| partition.reader_bytes)
        .sum::<usize>();
    let input_batches = streamed
        .iter()
        .map(|partition| partition.input_batches)
        .sum::<usize>();
    let compact_elapsed = streamed
        .iter()
        .map(|partition| partition.compact_elapsed)
        .sum::<Duration>();
    let partitions = streamed
        .into_iter()
        .map(|partition| partition.batches)
        .collect::<Vec<_>>();
    let output_batches = partitions.iter().map(Vec::len).sum::<usize>();
    let (reservation, retained_bytes, peak_reserved_bytes) = reservation.finish()?;
    debug_assert_eq!(retained_bytes, partition_physical_bytes(&partitions));
    if log_transfer_steps {
        eprintln!(
            "  [materialize-phase] mode=full-rows prepare_ms={:.3} stream_collect_ms={:.3} compact_cpu_ms={:.3}",
            prepare_elapsed.as_secs_f64() * 1000.0,
            collect_elapsed.as_secs_f64() * 1000.0,
            compact_elapsed.as_secs_f64() * 1000.0,
        );
        eprintln!(
            "  [materialize-stream] reader_bytes={} retained_bytes={} peak_reserved_bytes={} input_batches={} output_batches={}",
            reader_bytes, retained_bytes, peak_reserved_bytes, input_batches, output_batches,
        );
        eprintln!(
            "  [materialize-metrics]\n{}",
            DisplayableExecutionPlan::with_metrics(plan.as_ref()).indent(true)
        );
    }
    Ok(TransferHandoff::full_rows(
        partitions,
        input_row_hint,
        !filters.is_empty(),
        reservation,
    ))
}
