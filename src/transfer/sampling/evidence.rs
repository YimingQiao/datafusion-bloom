//! Materialized sample evidence and cardinality estimators.

use super::super::*;

/// Materialized rows plus the physical population they represent.
///
/// The rowset is used only for estimates. Formal execution always scans or
/// materializes exact rows after propagation decisions have been made.
#[derive(Debug)]
pub(in crate::transfer) struct SampledTable {
    pub(in crate::transfer) partitions: Vec<Vec<RecordBatch>>,
    pub(in crate::transfer) input_rows: usize,
    pub(in crate::transfer) output_rows: usize,
    /// Rows that can still match after conservative metadata pruning. `None`
    /// means the sampler did not establish a narrower source population.
    estimation_population_rows: Option<usize>,
    sampled_row_groups: usize,
    exact: bool,
}

impl SampledTable {
    pub(in crate::transfer) fn from_source_sample(
        partitions: Vec<Vec<RecordBatch>>,
        input_rows: usize,
        output_rows: usize,
    ) -> Self {
        Self {
            partitions,
            input_rows,
            output_rows,
            estimation_population_rows: None,
            sampled_row_groups: 0,
            exact: false,
        }
    }

    pub(in crate::transfer) fn from_output_partitions(partitions: Vec<Vec<RecordBatch>>) -> Self {
        let rows = count_rows(&partitions);
        Self {
            partitions,
            input_rows: rows,
            output_rows: rows,
            estimation_population_rows: None,
            sampled_row_groups: 0,
            exact: false,
        }
    }

    pub(super) fn from_candidate_sample(
        partitions: Vec<Vec<RecordBatch>>,
        input_rows: usize,
        output_rows: usize,
        population_rows: usize,
        sampled_row_groups: usize,
        exact: bool,
    ) -> Self {
        Self {
            partitions,
            input_rows,
            output_rows,
            estimation_population_rows: Some(population_rows),
            sampled_row_groups,
            exact,
        }
    }

    pub(in crate::transfer) fn estimate_local_rows(&self, base_rows: f64) -> f64 {
        if self.exact {
            return self.output_rows as f64;
        }
        if self.input_rows == 0 {
            return base_rows;
        }
        let population = self
            .estimation_population_rows
            .map_or(base_rows, |rows| rows as f64);
        (population * self.output_rows as f64 / self.input_rows as f64).clamp(0.0, base_rows)
    }

    pub(in crate::transfer) fn estimate_transfer_survivors(
        &self,
        survivors: usize,
        initial_rows: f64,
    ) -> f64 {
        if self.exact {
            return survivors as f64;
        }
        if self.output_rows == 0 {
            return initial_rows;
        }
        (initial_rows * survivors as f64 / self.output_rows as f64).clamp(0.0, initial_rows)
    }

    pub(in crate::transfer) fn sampled_row_groups(&self) -> usize {
        self.sampled_row_groups
    }

    pub(in crate::transfer) fn is_exact(&self) -> bool {
        self.exact
    }

    pub(in crate::transfer) fn one_observation_estimate(&self, base_rows: usize) -> f64 {
        let population = self
            .estimation_population_rows
            .unwrap_or(base_rows)
            .min(base_rows);
        (population as f64 / self.input_rows.max(1) as f64).max(1.0)
    }

    pub(in crate::transfer) fn estimation_population_rows(&self) -> Option<usize> {
        self.estimation_population_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_estimate_uses_the_candidate_population() {
        let sample = SampledTable::from_candidate_sample(vec![], 10_000, 100, 20_000, 4, false);
        assert_eq!(sample.estimate_local_rows(1_000_000.0), 200.0);
        assert_eq!(sample.one_observation_estimate(1_000_000), 2.0);
    }

    #[test]
    fn fully_pruned_population_is_an_exact_empty_sample() {
        let sample = SampledTable::from_candidate_sample(vec![], 0, 0, 0, 0, true);
        assert_eq!(sample.estimate_local_rows(1_000_000.0), 0.0);
    }
}
