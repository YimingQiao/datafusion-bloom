//! Bloom transfer for Apache DataFusion.
//!
//! Bloom runs before normal query execution and prepares reduced table-operator
//! handoffs for DataFusion's stock join operators. Propagation sources normally
//! own compact, fully materialized rows; terminal destinations may instead keep
//! a direct scan with transfer membership attached to the Parquet reader.

mod collection;
mod compat;
mod config;
mod filter;
mod graph;
mod lineage;
mod planner;
mod transfer;

pub use collection::BloomCollection;
pub use config::{BloomConfig, HandoffPolicy, ParquetMembershipPlacement, SamplingMode};
pub use planner::{BloomQueryPlanner, install_bloom};
