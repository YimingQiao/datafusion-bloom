//! Session-scoped storage for reusable prepared samples.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::{Arc, Mutex};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use tokio::sync::OnceCell;

/// Keep the planner cache useful for long-lived sessions without allowing a
/// stream of unrelated table snapshots to retain memory forever.
const MAX_PREPARED_SAMPLES: usize = 128;

type SampleCell = Arc<OnceCell<Arc<PreparedSourceSample>>>;

/// Query-planner-owned cache of raw, query-independent table samples.
///
/// A cell is installed before asynchronous preparation begins, so concurrent
/// queries share one build instead of scanning the same immutable source in
/// parallel. Completed entries are bounded by a small LRU and their Arrow
/// buffers remain registered with DataFusion's memory pool.
#[derive(Debug, Default)]
pub(crate) struct PreparedSampleCache {
    entries: Mutex<CacheEntries>,
}

#[derive(Debug, Default)]
struct CacheEntries {
    samples: HashMap<String, SampleCell>,
    lru: VecDeque<String>,
}

#[derive(Debug)]
pub(crate) struct PreparedSourceSample {
    pub(crate) partitions: Vec<Vec<RecordBatch>>,
    pub(crate) schema: SchemaRef,
    pub(crate) input_rows: usize,
    _reservation: MemoryReservation,
}

impl PreparedSourceSample {
    /// Bind cached Arrow buffers to DataFusion's memory pool for the lifetime
    /// of the reusable, query-independent sample.
    pub(crate) fn try_new(
        partitions: Vec<Vec<RecordBatch>>,
        schema: SchemaRef,
        input_rows: usize,
        context: &TaskContext,
    ) -> Result<Self> {
        let byte_size = partitions
            .iter()
            .flatten()
            .flat_map(|batch| batch.columns())
            .map(|array| array.get_array_memory_size())
            .sum();
        let reservation =
            MemoryConsumer::new("BloomPreparedSample").register(context.memory_pool());
        reservation.try_grow(byte_size)?;
        Ok(Self {
            partitions,
            schema,
            input_rows,
            _reservation: reservation,
        })
    }
}

impl PreparedSampleCache {
    /// Return one prepared source snapshot per cache key. The installed
    /// `OnceCell` makes concurrent cache misses single-flight, so a long-lived
    /// session never scans the same immutable source twice in parallel.
    pub(crate) async fn get_or_try_init<F, Fut>(
        &self,
        key: String,
        initialize: F,
    ) -> Result<Arc<PreparedSourceSample>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<PreparedSourceSample>>,
    {
        let cell = self.cell(key)?;
        let prepared = cell
            .get_or_try_init(|| async { initialize().await.map(Arc::new) })
            .await?;
        Ok(Arc::clone(prepared))
    }

    /// Release reusable samples before falling back after resource exhaustion;
    /// query-local materializations are owned elsewhere and are unaffected.
    pub(crate) fn clear(&self) -> Result<()> {
        let mut entries = self.entries.lock().map_err(|_| poisoned())?;
        entries.samples.clear();
        entries.lru.clear();
        Ok(())
    }

    /// Admit a build before it starts and evict completed entries only. An
    /// in-flight entry may temporarily exceed the bound to preserve
    /// single-flight behavior.
    fn cell(&self, key: String) -> Result<SampleCell> {
        let mut entries = self.entries.lock().map_err(|_| poisoned())?;
        if let Some(cell) = entries.samples.get(&key).cloned() {
            touch(&mut entries.lru, &key);
            return Ok(cell);
        }

        // Do not evict an in-flight build. A brief capacity overshoot is safer
        // than allowing another query to start a duplicate source scan.
        while entries.samples.len() >= MAX_PREPARED_SAMPLES {
            let Some(candidate) = entries.lru.pop_front() else {
                break;
            };
            let initialized = entries
                .samples
                .get(&candidate)
                .is_some_and(|cell| cell.get().is_some());
            if initialized {
                entries.samples.remove(&candidate);
                break;
            }
            entries.lru.push_back(candidate);
            if entries.lru.iter().all(|candidate| {
                entries
                    .samples
                    .get(candidate)
                    .is_some_and(|cell| cell.get().is_none())
            }) {
                break;
            }
        }

        let cell = Arc::new(OnceCell::new());
        entries.samples.insert(key.clone(), Arc::clone(&cell));
        touch(&mut entries.lru, &key);
        Ok(cell)
    }
}

fn touch(lru: &mut VecDeque<String>, key: &str) {
    if let Some(index) = lru.iter().position(|candidate| candidate == key) {
        lru.remove(index);
    }
    lru.push_back(key.to_string());
}

fn poisoned() -> DataFusionError {
    DataFusionError::Internal("Bloom prepared-sample cache lock was poisoned".to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use datafusion::arrow::datatypes::Schema;
    use datafusion::execution::TaskContext;
    use futures::future::join_all;

    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_misses_build_one_sample() -> Result<()> {
        let cache = Arc::new(PreparedSampleCache::default());
        let builds = Arc::new(AtomicUsize::new(0));
        let context = Arc::new(TaskContext::default());
        let jobs = (0..16).map(|_| {
            let cache = Arc::clone(&cache);
            let builds = Arc::clone(&builds);
            let context = Arc::clone(&context);
            tokio::spawn(async move {
                cache
                    .get_or_try_init("same-source".to_string(), || async move {
                        builds.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        PreparedSourceSample::try_new(
                            vec![vec![]],
                            Arc::new(Schema::empty()),
                            0,
                            &context,
                        )
                    })
                    .await
            })
        });
        for result in join_all(jobs).await {
            result.expect("cache task")?;
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn completed_cache_is_bounded() -> Result<()> {
        let cache = PreparedSampleCache::default();
        let context = TaskContext::default();
        for index in 0..(MAX_PREPARED_SAMPLES + 8) {
            cache
                .get_or_try_init(format!("source-{index}"), || async {
                    PreparedSourceSample::try_new(
                        vec![vec![]],
                        Arc::new(Schema::empty()),
                        0,
                        &context,
                    )
                })
                .await?;
        }
        let entries = cache.entries.lock().map_err(|_| poisoned())?;
        assert_eq!(entries.samples.len(), MAX_PREPARED_SAMPLES);
        assert_eq!(entries.lru.len(), MAX_PREPARED_SAMPLES);
        Ok(())
    }
}
