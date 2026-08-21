use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray, StringViewArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{NullEquality, ScalarValue};
use datafusion::logical_expr::{JoinType, Operator};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::physical_plan::joins::{CrossJoinExec, HashJoinExec, PartitionMode};

use super::handoff::predicate_is_expensive;
use super::policy::{MaterializationFacts, MaterializationStrategy, choose_materialization};
use super::sampling::localize_range;
use super::{
    MaterializedPartitionBuilder, compact_materialized_partition, native_join_filter_coverage,
    observed_handoff_widths, partition_physical_bytes,
};
use crate::config::HandoffPolicy;

#[test]
fn native_join_filter_coverage_counts_collect_left_boundaries() {
    fn leaf() -> Arc<dyn ExecutionPlan> {
        Arc::new(EmptyExec::new(Arc::new(Schema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]))))
    }

    fn join(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        mode: PartitionMode,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(
            HashJoinExec::try_new(
                left,
                right,
                vec![(
                    Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>,
                    Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>,
                )],
                None,
                &JoinType::Inner,
                None,
                mode,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap(),
        )
    }

    let covered = join(leaf(), leaf(), PartitionMode::CollectLeft);
    let mixed = join(covered, leaf(), PartitionMode::Partitioned);
    let covered = join(mixed, leaf(), PartitionMode::CollectLeft);
    let plan = join(covered, leaf(), PartitionMode::CollectLeft);
    let coverage = native_join_filter_coverage(&plan).unwrap();
    assert_eq!(coverage.join_count, 4);
    assert_eq!(coverage.collect_left, 3);
    assert!(
        coverage.collect_left * 3 >= coverage.join_count * 2,
        "three of four boundaries should satisfy the parallel guard"
    );

    let mixed_kind = Arc::new(CrossJoinExec::new(plan, leaf())) as Arc<dyn ExecutionPlan>;
    assert_eq!(
        native_join_filter_coverage(&mixed_kind),
        None,
        "non-hash joins make native dynamic-filter coverage incomplete"
    );
}

#[test]
fn global_sample_ranges_are_only_localized_after_intersection() {
    assert_eq!(localize_range(&(5..10), 0, 4), None);
    assert_eq!(localize_range(&(0..4), 5, 10), None);
    assert_eq!(localize_range(&(5..10), 7, 12), Some(0..3));
    assert_eq!(localize_range(&(8..14), 7, 12), Some(1..5));
}

#[test]
fn observed_buffers_keep_compressible_wide_schema_on_full_rows() {
    let mut fields = vec![Field::new("id", DataType::Int64, false)];
    fields
        .extend((0..8).map(|index| Field::new(format!("payload_{index}"), DataType::Utf8, false)));
    let schema = Arc::new(Schema::new(fields));
    let rows = 10_000;
    let ids = Arc::new(Int64Array::from_iter_values(0..rows as i64)) as ArrayRef;
    let payload = Arc::new(StringArray::from(vec!["x"; rows])) as ArrayRef;
    let mut columns = vec![ids];
    columns.extend((0..8).map(|_| Arc::clone(&payload)));
    let sample = vec![vec![
        RecordBatch::try_new(Arc::clone(&schema), columns).unwrap(),
    ]];

    let (full_row_width, transfer_row_width) =
        observed_handoff_widths(Some(&sample), schema.as_ref(), &[0]);
    let decision = choose_materialization(
        HandoffPolicy::CostBasedRowLocations,
        MaterializationFacts {
            source_rows: 300_000,
            locally_filtered_rows: 300_000,
            expected_rows: 1,
            full_row_width,
            transfer_row_width,
            has_local_filter: false,
        },
    );
    assert_eq!(decision.strategy, MaterializationStrategy::FullRows);
}

#[test]
fn observed_width_does_not_charge_a_slice_for_its_backing_buffer() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let rows = 10_000;
    let ids = Arc::new(Int64Array::from_iter_values(0..rows as i64)) as ArrayRef;
    let payload = Arc::new(StringArray::from(vec!["0123456789abcdef"; rows])) as ArrayRef;
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, payload]).unwrap();
    let sample = vec![vec![batch.slice(rows - 1, 1)]];

    let (full_row_width, transfer_row_width) =
        observed_handoff_widths(Some(&sample), schema.as_ref(), &[0]);
    assert!(full_row_width < 64, "logical width was {full_row_width}");
    assert_eq!(transfer_row_width, 20);
}

#[test]
fn compact_materialization_releases_filtered_view_buffers() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "payload",
        DataType::Utf8View,
        false,
    )]));
    let values = (0..256)
        .map(|index| format!("{index:04}-{}", "x".repeat(1024)))
        .collect::<Vec<_>>();
    let payload = Arc::new(StringViewArray::from_iter_values(
        values.iter().map(String::as_str),
    )) as ArrayRef;
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![payload])
        .unwrap()
        .slice(17, 1);
    let before = partition_physical_bytes(&[vec![batch.clone()]]);

    let compacted = compact_materialized_partition(vec![batch], 65_536).unwrap();
    let after = partition_physical_bytes(std::slice::from_ref(&compacted));
    let payload = compacted[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap();

    assert_eq!(payload.value(0), values[17]);
    assert!(after * 10 < before, "before={before} after={after}");
}

#[test]
fn compact_materialization_fills_batches_across_input_boundaries() {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let make_batch = |start: i64| {
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(start..start + 5_000)) as ArrayRef],
        )
        .unwrap()
    };

    let mut builder = MaterializedPartitionBuilder::new(8_192);
    assert!(builder.push(make_batch(0)).unwrap().is_empty());
    assert!(builder.buffered_physical_bytes() > 0);
    let mut compacted = builder.push(make_batch(5_000)).unwrap();
    assert_eq!(compacted.len(), 1, "a full batch must stream out early");
    compacted.extend(builder.finish().unwrap());
    assert_eq!(
        compacted
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        vec![8_192, 1_808]
    );
    let first = compacted[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let second = compacted[1]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(first.value(8_191), 8_191);
    assert_eq!(second.value(0), 8_192);
}

#[test]
fn transfer_predicate_order_classifies_strings_after_numeric_columns() {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]);
    let numeric = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("id", 0)),
        Operator::Eq,
        Arc::new(Literal::new(ScalarValue::Int64(Some(42)))),
    )) as Arc<dyn PhysicalExpr>;
    let string = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("label", 1)),
        Operator::Eq,
        Arc::new(Literal::new(ScalarValue::Utf8(Some("Bloom".to_string())))),
    )) as Arc<dyn PhysicalExpr>;

    assert!(!predicate_is_expensive(&numeric, &schema));
    assert!(predicate_is_expensive(&string, &schema));
}
