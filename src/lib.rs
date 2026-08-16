//! Bloom transfer for Apache DataFusion.
//!
//! Bloom runs before normal query execution and prepares reduced table-operator
//! handoffs for DataFusion's stock join operators. Propagation sources normally
//! own compact, fully materialized rows; terminal destinations may instead keep
//! a direct scan with transfer membership attached to the Parquet reader.

mod collection;
mod config;
mod filter;
mod graph;
mod handoff;
mod late_materialization;
mod lineage;
mod planner;
mod samples;
mod transfer;

pub use collection::BloomCollection;
pub use config::{BloomConfig, HandoffPolicy, ParquetMembershipPlacement};
pub use planner::{BloomQueryPlanner, install_bloom};
