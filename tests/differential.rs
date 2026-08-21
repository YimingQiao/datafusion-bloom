use std::sync::Arc;

use datafusion::arrow::array::{Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::common::test_util::batches_to_sort_string;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_bloom::{BloomConfig, install_bloom};

const QUERY: &str = "\
    SELECT a.payload AS ap, b.payload AS bp, c.payload AS cp \
    FROM table_a a \
    JOIN table_b b ON a.k1 = b.k1 AND a.k2 = b.k2 \
    JOIN table_c c ON b.k1 = c.k1 AND b.k2 = c.k2";

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int64, true),
        Field::new("k2", DataType::Utf8, true),
        Field::new("payload", DataType::Int64, false),
    ]))
}

fn make_partitions(seed: u64, payload_base: i64) -> Result<Vec<Vec<RecordBatch>>> {
    let schema = schema();
    let mut random = Lcg(seed);
    let mut keys_1 = Vec::with_capacity(42);
    let mut keys_2 = Vec::with_capacity(42);
    let mut payloads = Vec::with_capacity(42);
    let key_strings = ["a", "b", "c", "d", "e", "f"];

    for row in 0..42_i64 {
        let value = random.next();
        keys_1.push((!value.is_multiple_of(11)).then_some((value % 17) as i64));
        let value = random.next();
        keys_2.push((!value.is_multiple_of(13)).then_some(key_strings[(value % 6) as usize]));
        payloads.push(payload_base + row);
    }

    let mut partitions = vec![];
    for partition in 0..3 {
        let start = partition * 14;
        let end = start + 14;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(keys_1[start..end].to_vec())),
                Arc::new(StringArray::from(keys_2[start..end].to_vec())),
                Arc::new(Int64Array::from(payloads[start..end].to_vec())),
            ],
        )?;
        partitions.push(vec![batch]);
    }
    Ok(partitions)
}

fn context(bloom: bool) -> Result<SessionContext> {
    // Exercise the Bloom execution path rather than its parallel native-filter
    // bypass; the baseline uses the same physical parallelism.
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(1))
        .build();
    let state = if bloom {
        install_bloom(
            state,
            BloomConfig {
                excitation_threshold: 1.01,
                ..BloomConfig::default()
            },
        )?
    } else {
        state
    };
    Ok(SessionContext::new_with_state(state))
}

fn register(context: &SessionContext, name: &str, partitions: Vec<Vec<RecordBatch>>) -> Result<()> {
    context.register_table(name, Arc::new(MemTable::try_new(schema(), partitions)?))?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn randomized_nullable_composite_joins_match_datafusion() -> Result<()> {
    for seed in 0..12_u64 {
        let a = make_partitions(seed * 3 + 1, 1_000)?;
        let b = make_partitions(seed * 3 + 2, 2_000)?;
        let c = make_partitions(seed * 3 + 3, 3_000)?;

        let baseline = context(false)?;
        register(&baseline, "table_a", a.clone())?;
        register(&baseline, "table_b", b.clone())?;
        register(&baseline, "table_c", c.clone())?;

        let bloom = context(true)?;
        register(&bloom, "table_a", a)?;
        register(&bloom, "table_b", b)?;
        register(&bloom, "table_c", c)?;

        let expected = baseline.sql(QUERY).await?.collect().await?;
        let actual = bloom.sql(QUERY).await?.collect().await?;
        assert_eq!(
            batches_to_sort_string(&actual),
            batches_to_sort_string(&expected),
            "differential failure for seed {seed}"
        );
    }
    Ok(())
}
