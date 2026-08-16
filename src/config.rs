use datafusion::common::{Result, plan_err};

/// Representation retained between Bloom transfer and formal execution.
///
/// `FullRows` is Bloom's normal semantics: execute each participating table
/// operator once and retain every query-visible column. The experimental
/// cost-based policy may retain Parquet row locations instead, but it never
/// changes which tables participate in transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoffPolicy {
    FullRows,
    CostBasedRowLocations,
}

/// Where Parquet evaluates transfer membership for FullRows and Direct scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParquetMembershipPlacement {
    /// Evaluate membership in the Parquet row filter so payload decoding can
    /// follow the surviving row selection.
    Reader,
    /// Decode the scan projection first and evaluate membership in Arrow.
    /// This is retained as a diagnostic fallback for reader regressions.
    PostScan,
}

/// Configuration for the Bloom transfer planner.
#[derive(Debug, Clone)]
pub struct BloomConfig {
    /// Enable Bloom transfer.
    pub enabled: bool,
    /// Emit transfer scheduling and phase timing diagnostics to stderr.
    pub log_transfer_steps: bool,
    /// Desired false-positive rate for temporary Bloom filters.
    pub false_positive_rate: f64,
    /// Restrict the initial implementation to immutable in-memory sources.
    pub memory_sources_only: bool,
    /// Run DataFusion's physical optimizer again after materialization.
    ///
    /// This is experimental. The default preserves the join tree and build /
    /// probe choices already made from the original source statistics. Exact
    /// handoff row counts alone do not describe join-key multiplicity and can
    /// make a second join-selection pass substantially worse.
    pub reoptimize: bool,
    /// Select how a transfer materialization is handed to formal execution.
    ///
    /// This policy is deliberately independent of transfer scheduling.
    pub handoff_policy: HandoffPolicy,
    /// Placement of row-level membership within a Parquet materialization.
    /// This is independent of both propagation scheduling and handoff shape.
    pub parquet_membership_placement: ParquetMembershipPlacement,
    /// Maximum fixed-point rounds over the join graph.
    ///
    /// Stopping early can only retain extra rows; it cannot change query results.
    pub max_transfer_rounds: usize,
    /// Target number of post-local-filter rows retained per table sample.
    pub sample_rows: usize,
    /// Reactivate a table when its estimated cardinality falls below this
    /// fraction of the last committed baseline.
    pub excitation_threshold: f64,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_transfer_steps: false,
            false_positive_rate: 0.01,
            memory_sources_only: true,
            reoptimize: false,
            handoff_policy: HandoffPolicy::FullRows,
            parquet_membership_placement: ParquetMembershipPlacement::Reader,
            max_transfer_rounds: 64,
            sample_rows: 10_000,
            excitation_threshold: 1.0,
        }
    }
}

impl BloomConfig {
    /// Emit detailed transfer-phase diagnostics.
    pub fn with_transfer_logging(mut self) -> Self {
        self.log_transfer_steps = true;
        self
    }

    /// Allow transfer over any bounded physical source.
    ///
    /// This is appropriate for immutable or otherwise repeatable bounded
    /// sources. The conservative default limits initial deployments to
    /// in-memory sources.
    pub fn with_all_bounded_sources(mut self) -> Self {
        self.memory_sources_only = false;
        self
    }

    /// Enable the experimental cost-based Parquet row-location handoff.
    ///
    /// Tables that do not satisfy the independent materialization cost policy
    /// continue to use `FullRows`.
    pub fn with_row_locations(mut self) -> Self {
        self.handoff_policy = HandoffPolicy::CostBasedRowLocations;
        self
    }

    /// Keep transfer membership above the Parquet reader.
    ///
    /// This is primarily useful for profiling scans where reader-side row
    /// selection amplifies decoding or predicate-cache work.
    pub fn with_post_scan_membership(mut self) -> Self {
        self.parquet_membership_placement = ParquetMembershipPlacement::PostScan;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if !(0.0..1.0).contains(&self.false_positive_rate) {
            return plan_err!("Bloom false_positive_rate must be greater than 0 and less than 1");
        }
        if self.max_transfer_rounds == 0 {
            return plan_err!("Bloom max_transfer_rounds must be greater than 0");
        }
        if self.sample_rows == 0 {
            return plan_err!("Bloom sample_rows must be greater than 0");
        }
        if !self.excitation_threshold.is_finite() || self.excitation_threshold <= 0.0 {
            return plan_err!("Bloom excitation_threshold must be finite and greater than 0");
        }
        Ok(())
    }
}
