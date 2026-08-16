use std::collections::{BTreeMap, BTreeSet};

use datafusion::common::{Result, internal_err};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::Column;

use crate::graph::{BloomGraph, TableId};

/// Identity of one join-key column in the transfer graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ColumnId {
    pub(crate) table: TableId,
    pub(crate) column: usize,
}

/// Lineage captured when an outgoing transfer structure is built.
///
/// `columns` drives subsumption, `lineages` identifies the contributing
/// rowsets, and `key_columns` distinguishes filters built from different key
/// vectors after their propagated column sets happen to converge.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct LineageSnapshot {
    columns: BTreeSet<ColumnId>,
    lineages: BTreeSet<TableId>,
    key_columns: Vec<ColumnId>,
}

impl LineageSnapshot {
    pub(crate) fn is_subset_of(&self, other: &Self) -> bool {
        self.columns.is_subset(&other.columns)
    }
}

/// Per-join-key lineage state for one Bloom transfer phase.
///
/// A transfer restricts an entire destination rowset. Consequently, source-key
/// lineage is attached to every join key of the destination, not only to the
/// key on which the incoming membership test was evaluated.
#[derive(Debug)]
pub(crate) struct LineageTracker {
    per_table: Vec<BTreeSet<TableId>>,
    join_columns: Vec<BTreeSet<usize>>,
    per_column: BTreeMap<ColumnId, BTreeSet<ColumnId>>,
}

impl LineageTracker {
    pub(crate) fn try_new(graph: &BloomGraph) -> Result<Self> {
        let mut tracker = Self {
            per_table: (0..graph.tables.len())
                .map(|table| BTreeSet::from([table]))
                .collect(),
            join_columns: vec![BTreeSet::new(); graph.tables.len()],
            per_column: BTreeMap::new(),
        };

        for edge in &graph.edges {
            tracker.seed_columns(edge.left, &edge.left_keys)?;
            tracker.seed_columns(edge.right, &edge.right_keys)?;
        }
        Ok(tracker)
    }

    pub(crate) fn edge_carries_new_info(
        &self,
        source: TableId,
        destination: TableId,
        source_keys: &[std::sync::Arc<dyn PhysicalExpr>],
        destination_keys: &[std::sync::Arc<dyn PhysicalExpr>],
    ) -> Result<bool> {
        if source_keys.len() != destination_keys.len() {
            return internal_err!(
                "Bloom edge has {} source keys and {} destination keys",
                source_keys.len(),
                destination_keys.len()
            );
        }

        for (source_key, destination_key) in source_keys.iter().zip(destination_keys) {
            let source_column = column_id(source, source_key)?;
            let destination_column = column_id(destination, destination_key)?;
            let Some(source_lineage) = self.per_column.get(&source_column) else {
                continue;
            };
            let Some(destination_lineage) = self.per_column.get(&destination_column) else {
                return Ok(true);
            };
            if !source_lineage.is_subset(destination_lineage) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn propagate(
        &mut self,
        source: TableId,
        destination: TableId,
        source_keys: &[std::sync::Arc<dyn PhysicalExpr>],
    ) -> Result<()> {
        let source_tables = self.per_table.get(source).cloned().ok_or_else(|| {
            datafusion::common::DataFusionError::Internal(format!(
                "Bloom lineage references unknown source table {source}"
            ))
        })?;
        let destination_tables = self.per_table.get_mut(destination).ok_or_else(|| {
            datafusion::common::DataFusionError::Internal(format!(
                "Bloom lineage references unknown destination table {destination}"
            ))
        })?;
        destination_tables.extend(source_tables);

        let source_columns = self.union_column_lineage(source, source_keys)?;
        let destination_join_columns = self
            .join_columns
            .get(destination)
            .cloned()
            .unwrap_or_default();
        for column in destination_join_columns {
            self.per_column
                .entry(ColumnId {
                    table: destination,
                    column,
                })
                .or_default()
                .extend(source_columns.iter().copied());
        }
        Ok(())
    }

    pub(crate) fn snapshot(
        &self,
        source: TableId,
        source_keys: &[std::sync::Arc<dyn PhysicalExpr>],
    ) -> Result<LineageSnapshot> {
        Ok(LineageSnapshot {
            columns: self.union_column_lineage(source, source_keys)?,
            lineages: self.per_table.get(source).cloned().unwrap_or_default(),
            key_columns: source_keys
                .iter()
                .map(|key| column_id(source, key))
                .collect::<Result<_>>()?,
        })
    }

    fn seed_columns(
        &mut self,
        table: TableId,
        keys: &[std::sync::Arc<dyn PhysicalExpr>],
    ) -> Result<()> {
        for key in keys {
            let column = column_id(table, key)?;
            self.join_columns[table].insert(column.column);
            self.per_column.entry(column).or_insert_with(|| {
                let mut lineage = BTreeSet::new();
                lineage.insert(column);
                lineage
            });
        }
        Ok(())
    }

    fn union_column_lineage(
        &self,
        table: TableId,
        keys: &[std::sync::Arc<dyn PhysicalExpr>],
    ) -> Result<BTreeSet<ColumnId>> {
        let mut output = BTreeSet::new();
        for key in keys {
            let column = column_id(table, key)?;
            if let Some(lineage) = self.per_column.get(&column) {
                output.extend(lineage.iter().copied());
            }
        }
        Ok(output)
    }
}

fn column_id(table: TableId, expression: &std::sync::Arc<dyn PhysicalExpr>) -> Result<ColumnId> {
    let Some(column) = expression.downcast_ref::<Column>() else {
        return internal_err!("Bloom lineage requires column join keys");
    };
    Ok(ColumnId {
        table,
        column: column.index(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_expr::expressions::Column;

    use super::{ColumnId, LineageTracker};

    fn key(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new(name, index))
    }

    fn tracker() -> LineageTracker {
        LineageTracker {
            per_table: (0..3).map(|table| [table].into_iter().collect()).collect(),
            join_columns: vec![
                [0].into_iter().collect(),
                [0, 1].into_iter().collect(),
                [0].into_iter().collect(),
            ],
            per_column: BTreeMap::from([
                (
                    ColumnId {
                        table: 0,
                        column: 0,
                    },
                    [ColumnId {
                        table: 0,
                        column: 0,
                    }]
                    .into_iter()
                    .collect(),
                ),
                (
                    ColumnId {
                        table: 1,
                        column: 0,
                    },
                    [ColumnId {
                        table: 1,
                        column: 0,
                    }]
                    .into_iter()
                    .collect(),
                ),
                (
                    ColumnId {
                        table: 1,
                        column: 1,
                    },
                    [ColumnId {
                        table: 1,
                        column: 1,
                    }]
                    .into_iter()
                    .collect(),
                ),
                (
                    ColumnId {
                        table: 2,
                        column: 0,
                    },
                    [ColumnId {
                        table: 2,
                        column: 0,
                    }]
                    .into_iter()
                    .collect(),
                ),
            ]),
        }
    }

    use std::collections::BTreeMap;

    #[test]
    fn transfer_marks_every_destination_join_key() {
        let mut tracker = tracker();
        tracker.propagate(0, 1, &[key("a", 0)]).unwrap();

        assert!(
            tracker
                .edge_carries_new_info(1, 2, &[key("b_other", 1)], &[key("c", 0)])
                .unwrap()
        );
        let snapshot = tracker.snapshot(1, &[key("b_other", 1)]).unwrap();
        assert!(snapshot.columns.contains(&ColumnId {
            table: 0,
            column: 0
        }));
        assert!(snapshot.columns.contains(&ColumnId {
            table: 1,
            column: 1
        }));
    }

    #[test]
    fn snapshot_subsumption_is_column_granular() {
        let mut tracker = tracker();
        let before = tracker.snapshot(1, &[key("b", 1)]).unwrap();
        tracker.propagate(0, 1, &[key("a", 0)]).unwrap();
        let after = tracker.snapshot(1, &[key("b", 1)]).unwrap();
        assert!(before.is_subset_of(&after));
        assert!(!after.is_subset_of(&before));
    }
}
