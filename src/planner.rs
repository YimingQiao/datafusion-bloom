use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::Result;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SessionState, SessionStateBuilder};
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};

use crate::compat::{is_recoverable_transfer_error, is_resource_exhausted};
use crate::config::BloomConfig;
use crate::transfer::{BloomTransferEngine, PreparedSampleCache, RowGroupLayoutCache};

/// DataFusion query planner that inserts Bloom's transfer phase between P0 and P1.
#[derive(Debug, Clone)]
pub struct BloomQueryPlanner {
    config: BloomConfig,
    samples: Arc<PreparedSampleCache>,
    row_group_layouts: Arc<RowGroupLayoutCache>,
}

impl BloomQueryPlanner {
    /// Create a session-scoped planner. Prepared source data is intentionally
    /// shared across queries, while every transfer graph remains query-scoped.
    pub fn new(config: BloomConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            samples: Arc::new(PreparedSampleCache::default()),
            row_group_layouts: Arc::new(RowGroupLayoutCache::default()),
        })
    }

    pub fn config(&self) -> &BloomConfig {
        &self.config
    }
}

/// Return a copy of `state` whose query planner runs Bloom transfer.
pub fn install_bloom(state: SessionState, config: BloomConfig) -> Result<SessionState> {
    let planner = Arc::new(BloomQueryPlanner::new(config)?);
    Ok(SessionStateBuilder::new_from_existing(state)
        .with_query_planner(planner)
        .build())
}

#[async_trait]
impl QueryPlanner for BloomQueryPlanner {
    /// Plan P0 and the formal query separately, run transfer between them, and
    /// replace only table-operator leaves for which transfer produced a safe
    /// handoff. Recoverable transfer failures abandon the whole rewrite so the
    /// untouched native DataFusion plan remains the correctness fallback.
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let default_planner = DefaultPhysicalPlanner::default();

        if !self.config.enabled {
            return default_planner
                .create_physical_plan(logical_plan, session_state)
                .await;
        }

        // Keep DataFusion's native physical plan as the formal join plan. Bloom
        // only replaces table-operator leaves that produce a transfer handoff;
        // every untouched leaf therefore retains DataFusion's runtime dynamic
        // filters and the join stage remains otherwise unchanged.
        let formal_plan = default_planner
            .create_physical_plan(logical_plan, session_state)
            .await?;

        // A table-operator child must be independently executable during
        // transfer. Runtime state produced by a surrounding join cannot be part
        // of P0, so build a separate P0 with dynamic-filter mechanisms disabled.
        let mut p0_state = session_state.clone();
        let optimizer = &mut p0_state.config_mut().options_mut().optimizer;
        // Keep DataFusion's already selected join tree and build/probe sides.
        // Bloom rewrites table-operator leaves after the native optimizer, as
        // the DuckDB extension does. Reconstructing join order from exact
        // handoff row counts is unsafe: ordinary NDV statistics can badly
        // underestimate a duplicate-heavy intermediate relation.
        optimizer.enable_dynamic_filter_pushdown = false;
        optimizer.enable_join_dynamic_filter_pushdown = false;
        optimizer.enable_topk_dynamic_filter_pushdown = false;
        optimizer.enable_aggregate_dynamic_filter_pushdown = false;

        let p0 = default_planner
            .create_physical_plan(logical_plan, &p0_state)
            .await?;
        let transfer = BloomTransferEngine::new(
            self.config.clone(),
            Arc::clone(&self.samples),
            Arc::clone(&self.row_group_layouts),
        );
        let rewritten = match transfer
            .rewrite(p0, Arc::clone(&formal_plan), session_state.task_ctx())
            .await
        {
            Ok(rewritten) => rewritten,
            Err(error) if is_recoverable_transfer_error(&error) => {
                if is_resource_exhausted(&error) {
                    self.samples.clear()?;
                }
                if self.config.log_transfer_steps {
                    eprintln!("[Bloom] fallback=native reason={error}");
                }
                formal_plan
            }
            Err(error) => return Err(error),
        };

        if self.config.reoptimize {
            default_planner.optimize_physical_plan(rewritten, session_state, |_, _| {})
        } else {
            Ok(rewritten)
        }
    }
}
