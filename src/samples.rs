use std::collections::HashMap;
use std::sync::Mutex;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result};

/// Query-planner-owned cache of raw, query-independent table samples.
///
/// The planner is shared by every query in a `SessionContext`, so a prepared
/// sample built during workload warmup is reused by later physical plans.
#[derive(Debug, Default)]
pub(crate) struct PreparedSampleCache {
    entries: Mutex<HashMap<String, PreparedSourceSample>>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedSourceSample {
    pub(crate) partitions: Vec<Vec<RecordBatch>>,
    pub(crate) schema: SchemaRef,
    pub(crate) input_rows: usize,
}

impl PreparedSampleCache {
    pub(crate) fn get(&self, key: &str) -> Result<Option<PreparedSourceSample>> {
        let entries = self.entries.lock().map_err(|_| {
            DataFusionError::Internal("Bloom prepared-sample cache lock was poisoned".to_string())
        })?;
        Ok(entries.get(key).cloned())
    }

    pub(crate) fn insert(&self, key: String, sample: PreparedSourceSample) -> Result<()> {
        let mut entries = self.entries.lock().map_err(|_| {
            DataFusionError::Internal("Bloom prepared-sample cache lock was poisoned".to_string())
        })?;
        entries.entry(key).or_insert(sample);
        Ok(())
    }
}
