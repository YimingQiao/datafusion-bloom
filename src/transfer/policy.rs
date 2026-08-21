//! Physical handoff policy, deliberately independent from propagation.

use datafusion::arrow::datatypes::{DataType, Schema};

use crate::config::HandoffPolicy;

// This policy is deliberately independent from propagation scheduling. It
// chooses how an already selected transfer source crosses into formal
// execution; it never decides whether information should continue propagating.
// Row locations have fixed setup and random-access costs that a simple row
// count ratio does not capture, so the experimental path is conservative.
const MIN_FULL_PAYLOAD_BYTES: u128 = 64 * 1024 * 1024;
const MIN_WIDTH_SAVING_BYTES: usize = 192;
const MIN_WIDTH_RATIO: usize = 6;
const MIN_TRANSFER_REDUCTION: usize = 32;
const MIN_LOCAL_FILTER_REDUCTION: usize = 8;
const MAX_SELECTED_ROWS: usize = 250_000;
const REQUIRED_COST_ADVANTAGE: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MaterializationStrategy {
    /// Read and own every query-required column at the transfer boundary.
    FullRows,
    /// Retain stable source positions and defer wide payload columns.
    RowLocations,
    /// Discover positions with local columns before reading transfer keys.
    TwoPassRowLocations,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MaterializationFacts {
    pub(crate) source_rows: usize,
    pub(crate) locally_filtered_rows: usize,
    pub(crate) expected_rows: usize,
    pub(crate) full_row_width: usize,
    pub(crate) transfer_row_width: usize,
    pub(crate) has_local_filter: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MaterializationDecision {
    pub(crate) strategy: MaterializationStrategy,
    pub(crate) reason: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RowLocationLocality {
    pub(crate) selected_rows: usize,
    pub(crate) contiguous_runs: usize,
    pub(crate) touched_row_groups: usize,
    pub(crate) total_row_groups: usize,
    pub(crate) touched_row_group_rows: usize,
}

/// Choose only the physical representation of a transfer handoff.
///
/// FullRows is Bloom's default semantics. Late materialization is admitted
/// only when the estimated I/O saving is large enough to pay for a second
/// Parquet access; observed row locality is validated separately after the
/// candidate locations are known.
pub(crate) fn choose_materialization(
    policy: HandoffPolicy,
    facts: MaterializationFacts,
) -> MaterializationDecision {
    if policy == HandoffPolicy::FullRows {
        return full_rows("row_locations_disabled");
    }
    if facts.source_rows == 0 || facts.locally_filtered_rows == 0 {
        return full_rows("empty_or_unknown_source");
    }
    if facts.expected_rows == 0 {
        // An exact empty FullRows handoff avoids retaining locations and avoids
        // a formal Parquet scan altogether.
        return full_rows("expected_empty");
    }
    if facts.expected_rows > MAX_SELECTED_ROWS {
        return full_rows("too_many_selected_rows");
    }
    if facts.expected_rows.saturating_mul(MIN_TRANSFER_REDUCTION) > facts.locally_filtered_rows {
        return full_rows("transfer_not_selective_enough");
    }
    if facts.full_row_width < facts.transfer_row_width.saturating_mul(MIN_WIDTH_RATIO)
        || facts
            .full_row_width
            .saturating_sub(facts.transfer_row_width)
            < MIN_WIDTH_SAVING_BYTES
    {
        return full_rows("table_not_wide_enough");
    }

    let full_payload = facts.locally_filtered_rows as u128 * facts.full_row_width as u128;
    if full_payload < MIN_FULL_PAYLOAD_BYTES {
        return full_rows("full_payload_too_small");
    }
    let row_location_payload = facts.locally_filtered_rows as u128
        * facts.transfer_row_width as u128
        + facts.expected_rows as u128 * facts.full_row_width as u128;
    if row_location_payload.saturating_mul(REQUIRED_COST_ADVANTAGE as u128) >= full_payload {
        return full_rows("insufficient_projected_saving");
    }

    let two_pass = facts.has_local_filter
        && facts
            .locally_filtered_rows
            .saturating_mul(MIN_LOCAL_FILTER_REDUCTION)
            <= facts.source_rows;
    MaterializationDecision {
        strategy: if two_pass {
            MaterializationStrategy::TwoPassRowLocations
        } else {
            MaterializationStrategy::RowLocations
        },
        reason: "projected_scan_saving",
    }
}

fn full_rows(reason: &'static str) -> MaterializationDecision {
    MaterializationDecision {
        strategy: MaterializationStrategy::FullRows,
        reason,
    }
}

/// Apply the second, storage-aware gate for late materialization.
///
/// A selective result can still be expensive when it touches most row groups.
/// This rejects handoffs whose random access and decode amplification would
/// likely cost more than materializing FullRows once.
pub(crate) fn row_locations_are_concentrated(locality: RowLocationLocality) -> bool {
    if locality.selected_rows == 0 {
        return false;
    }

    // A small absolute number of seeks is cheap even when those rows are in
    // different groups. For larger selections require either useful runs or a
    // small row-group decode envelope, and reject extreme decode amplification.
    let few_runs = locality.contiguous_runs <= 64;
    let useful_runs = locality.contiguous_runs.saturating_mul(8) <= locality.selected_rows;
    let clustered_groups = locality.total_row_groups > 0
        && locality.touched_row_groups.saturating_mul(4) <= locality.total_row_groups;
    let bounded_decode_envelope = locality.selected_rows <= 64
        || locality.touched_row_group_rows <= locality.selected_rows.saturating_mul(256);
    (few_runs || (useful_runs && clustered_groups)) && bounded_decode_envelope
}

pub(crate) fn estimated_schema_width(schema: &Schema) -> usize {
    schema
        .fields()
        .iter()
        .map(|field| estimated_type_width(field.data_type()))
        .sum::<usize>()
        .max(1)
}

pub(crate) fn estimated_projection_width(schema: &Schema, columns: &[usize]) -> usize {
    columns
        .iter()
        .map(|&index| estimated_type_width(schema.field(index).data_type()))
        // Stable file and row identifiers are part of a row-location handoff.
        .sum::<usize>()
        .saturating_add(12)
        .max(1)
}

pub(crate) fn estimated_type_width(data_type: &DataType) -> usize {
    match data_type {
        DataType::Null => 0,
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 | DataType::Float16 => 2,
        DataType::Int32
        | DataType::UInt32
        | DataType::Float32
        | DataType::Date32
        | DataType::Time32(_) => 4,
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_)
        | DataType::Interval(_) => 8,
        DataType::Decimal32(_, _) => 4,
        DataType::Decimal64(_, _) => 8,
        DataType::Decimal128(_, _) => 16,
        DataType::Decimal256(_, _) => 32,
        DataType::Utf8 | DataType::Binary | DataType::Utf8View | DataType::BinaryView => 32,
        DataType::LargeUtf8 | DataType::LargeBinary => 40,
        DataType::FixedSizeBinary(size) => usize::try_from(*size).unwrap_or(64),
        DataType::Dictionary(key, _) => estimated_type_width(key),
        DataType::List(_) | DataType::ListView(_) | DataType::Map(_, _) => 48,
        DataType::LargeList(_) | DataType::LargeListView(_) => 56,
        DataType::FixedSizeList(field, size) => estimated_type_width(field.data_type())
            .saturating_mul(usize::try_from(*size).unwrap_or(1)),
        DataType::Struct(fields) => fields
            .iter()
            .map(|field| estimated_type_width(field.data_type()))
            .sum(),
        DataType::Union(fields, _) => fields
            .iter()
            .map(|(_, field)| estimated_type_width(field.data_type()))
            .max()
            .unwrap_or(0)
            .saturating_add(1),
        DataType::RunEndEncoded(_, values) => estimated_type_width(values.data_type()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn useful_wide_facts() -> MaterializationFacts {
        MaterializationFacts {
            source_rows: 20_000_000,
            locally_filtered_rows: 10_000_000,
            expected_rows: 10_000,
            full_row_width: 256,
            transfer_row_width: 24,
            has_local_filter: false,
        }
    }

    #[test]
    fn full_rows_policy_never_selects_locations() {
        let decision = choose_materialization(HandoffPolicy::FullRows, useful_wide_facts());
        assert_eq!(decision.strategy, MaterializationStrategy::FullRows);
    }

    #[test]
    fn cost_policy_rejects_narrow_tables() {
        let mut facts = useful_wide_facts();
        facts.full_row_width = 48;
        let decision = choose_materialization(HandoffPolicy::CostBasedRowLocations, facts);
        assert_eq!(decision.strategy, MaterializationStrategy::FullRows);
        assert_eq!(decision.reason, "table_not_wide_enough");
    }

    #[test]
    fn cost_policy_requires_transfer_selectivity() {
        let mut facts = useful_wide_facts();
        facts.expected_rows = 1_000_000;
        let decision = choose_materialization(HandoffPolicy::CostBasedRowLocations, facts);
        assert_eq!(decision.strategy, MaterializationStrategy::FullRows);
        assert_eq!(decision.reason, "too_many_selected_rows");
    }

    #[test]
    fn cost_policy_selects_locations_only_for_clear_savings() {
        let decision =
            choose_materialization(HandoffPolicy::CostBasedRowLocations, useful_wide_facts());
        assert_eq!(decision.strategy, MaterializationStrategy::RowLocations);
    }

    #[test]
    fn selective_local_filter_uses_two_pass_locations() {
        let mut facts = useful_wide_facts();
        facts.locally_filtered_rows = 1_000_000;
        facts.expected_rows = 1_000;
        facts.has_local_filter = true;
        let decision = choose_materialization(HandoffPolicy::CostBasedRowLocations, facts);
        assert_eq!(
            decision.strategy,
            MaterializationStrategy::TwoPassRowLocations
        );
    }

    #[test]
    fn locality_rejects_scattered_rows_across_the_file() {
        assert!(!row_locations_are_concentrated(RowLocationLocality {
            selected_rows: 2_000,
            contiguous_runs: 1_950,
            touched_row_groups: 60,
            total_row_groups: 60,
            touched_row_group_rows: 15_000_000,
        }));
    }

    #[test]
    fn locality_accepts_a_small_or_clustered_selection() {
        assert!(row_locations_are_concentrated(RowLocationLocality {
            selected_rows: 12,
            contiguous_runs: 12,
            touched_row_groups: 4,
            total_row_groups: 60,
            touched_row_group_rows: 1_000_000,
        }));
        assert!(row_locations_are_concentrated(RowLocationLocality {
            selected_rows: 20_000,
            contiguous_runs: 1_000,
            touched_row_groups: 5,
            total_row_groups: 60,
            touched_row_group_rows: 1_000_000,
        }));
    }
}
