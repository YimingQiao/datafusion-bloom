use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::source::{DataSource, DataSourceExec};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion::physical_plan::execution_plan::SchedulingType;
use datafusion::physical_plan::{
    DisplayFormatType, ExecutionPlan, SendableRecordBatchStream, Statistics,
};

/// Query-scoped, immutable materialized data produced by Bloom transfer.
pub struct BloomCollection {
    schema: SchemaRef,
    partitions: Vec<Vec<RecordBatch>>,
    input_row_count: usize,
    row_count: usize,
    byte_size: usize,
    generation: u64,
    _reservation: MemoryReservation,
}

impl fmt::Debug for BloomCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BloomCollection")
            .field("partitions", &self.partitions.len())
            .field("input_row_count", &self.input_row_count)
            .field("row_count", &self.row_count)
            .field("byte_size", &self.byte_size)
            .field("generation", &self.generation)
            .finish()
    }
}

impl BloomCollection {
    pub(crate) fn try_new(
        mut partitions: Vec<Vec<RecordBatch>>,
        schema: SchemaRef,
        input_row_count: usize,
        generation: u64,
        label: &str,
        context: &TaskContext,
        reservation: Option<MemoryReservation>,
    ) -> Result<Arc<Self>> {
        if partitions.is_empty() {
            partitions.push(vec![]);
        }

        for batch in partitions.iter().flatten() {
            if batch.schema() != schema {
                return datafusion::common::plan_err!(
                    "Bloom materialization schema mismatch: expected {}, got {}",
                    schema,
                    batch.schema()
                );
            }
        }

        let row_count = partitions.iter().flatten().map(RecordBatch::num_rows).sum();
        if row_count > input_row_count {
            return datafusion::common::internal_err!(
                "Bloom materialization grew from {input_row_count} to {row_count} rows"
            );
        }
        let byte_size = partitions
            .iter()
            .flatten()
            .flat_map(|batch| batch.columns())
            .map(|array| array.get_array_memory_size())
            .sum();

        let reservation = if let Some(reservation) = reservation {
            reservation.try_resize(byte_size)?;
            reservation
        } else {
            let reservation = MemoryConsumer::new(format!("BloomCollection[{label}]"))
                .register(context.memory_pool());
            reservation.try_grow(byte_size)?;
            reservation
        };

        Ok(Arc::new(Self {
            schema,
            partitions,
            input_row_count,
            row_count,
            byte_size,
            generation,
            _reservation: reservation,
        }))
    }

    /// Schema of the materialized table operator.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Partitioned Arrow batches retained by this collection.
    pub fn partitions(&self) -> &[Vec<RecordBatch>] {
        &self.partitions
    }

    /// Exact number of materialized rows.
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Exact row count before transfer filters were applied.
    pub fn input_row_count(&self) -> usize {
        self.input_row_count
    }

    /// Accounted Arrow array memory in bytes.
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }

    /// Transfer generation represented by this materialization.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn into_exec(
        self: Arc<Self>,
        label: impl Into<String>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let memory = MemorySourceConfig::try_new(&self.partitions, Arc::clone(&self.schema), None)?;
        let source = BloomCollectionSource {
            collection: self,
            memory,
            label: label.into(),
        };
        Ok(Arc::new(DataSourceExec::new(Arc::new(source))))
    }
}

#[derive(Clone)]
pub(crate) struct BloomCollectionSource {
    collection: Arc<BloomCollection>,
    memory: MemorySourceConfig,
    label: String,
}

impl fmt::Debug for BloomCollectionSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BloomCollectionSource")
            .field("label", &self.label)
            .field("collection", &self.collection)
            .finish()
    }
}

impl DataSource for BloomCollectionSource {
    fn open(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.memory.open(partition, context)
    }

    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "BloomCollection: label={}, partitions={}, input_rows={}, rows={}, bytes={}, generation={}",
                self.label,
                self.collection.partitions.len(),
                self.collection.input_row_count,
                self.collection.row_count,
                self.collection.byte_size,
                self.collection.generation
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "format=BloomCollection")?;
                writeln!(f, "label={}", self.label)?;
                writeln!(f, "input_rows={}", self.collection.input_row_count)?;
                writeln!(f, "rows={}", self.collection.row_count)?;
                writeln!(f, "bytes={}", self.collection.byte_size)
            }
        }
    }

    fn output_partitioning(&self) -> Partitioning {
        self.memory.output_partitioning()
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        self.memory.eq_properties()
    }

    fn scheduling_type(&self) -> SchedulingType {
        self.memory.scheduling_type()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        self.memory.partition_statistics(partition)
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        Some(Arc::new(Self {
            collection: Arc::clone(&self.collection),
            memory: self.memory.clone().with_limit(limit),
            label: self.label.clone(),
        }))
    }

    fn fetch(&self) -> Option<usize> {
        self.memory.fetch()
    }

    fn try_swapping_with_projection(
        &self,
        _projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        // Keeping ProjectionExec above this source is always correct. Projection
        // pushdown can be added once collection ownership and accounting can be
        // preserved through source rewrites.
        Ok(None)
    }
}
