use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::time::{Duration, Instant};

use crate::buffer::BufferPool;
use crate::page::PageManager;
use crate::{PageId, Result, StorageEngineError, TransactionId, VectorRow};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VacuumStats {
    pub pages_processed: u64,
    pub rows_cleaned: u64,
    pub space_reclaimed: u64,
    pub duration_ms: u64,
    pub vacuum_type: VacuumType,
    pub started_at: u64,
    pub completed_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VacuumType {
    Auto,
    Manual,
    Analyze,
    Full,
}

#[derive(Debug)]
struct PageVacuumInfo {
    page_id: PageId,
    dead_rows: usize,
    free_space: usize,
    last_vacuum: Option<Instant>,
    priority_score: u64,
}

pub struct VacuumManager {
    stats: RwLock<HashMap<VacuumType, VacuumStats>>,
    auto_vacuum_enabled: bool,
    vacuum_threshold: f32,
    vacuum_scale_factor: f32,
    last_vacuum_time: RwLock<HashMap<PageId, Instant>>,
    vacuum_cost_budget: AtomicU64,
    vacuum_cost_limit: u64,
}

impl VacuumManager {
    pub fn new() -> Self {
        Self {
            stats: RwLock::new(HashMap::new()),
            auto_vacuum_enabled: true,
            vacuum_threshold: 0.2,    // 20% dead tuples
            vacuum_scale_factor: 0.1, // 10% scaling factor
            last_vacuum_time: RwLock::new(HashMap::new()),
            vacuum_cost_budget: AtomicU64::new(0),
            vacuum_cost_limit: 200, // Cost-based vacuum delay
        }
    }

    pub async fn run(
        &self,
        buffer_pool: &Arc<BufferPool>,
        page_manager: &Arc<PageManager>,
    ) -> Result<VacuumStats> {
        self.run_with_type(buffer_pool, page_manager, VacuumType::Auto)
            .await
    }

    pub async fn run_with_type(
        &self,
        buffer_pool: &Arc<BufferPool>,
        page_manager: &Arc<PageManager>,
        vacuum_type: VacuumType,
    ) -> Result<VacuumStats> {
        let start_time = Instant::now();
        let started_at = current_timestamp();

        let mut pages_processed = 0;
        let mut rows_cleaned = 0;
        let mut space_reclaimed = 0;

        let pages_to_vacuum = self
            .identify_vacuum_candidates(buffer_pool, vacuum_type)
            .await?;

        for page_info in pages_to_vacuum {
            if self.should_throttle_vacuum().await {
                tokio::task::yield_now().await;
                continue;
            }

            let vacuum_result = self.vacuum_page(buffer_pool, page_info.page_id).await?;

            pages_processed += 1;
            rows_cleaned += vacuum_result.rows_cleaned;
            space_reclaimed += vacuum_result.space_reclaimed;

            self.update_vacuum_cost(10).await;
            self.record_page_vacuum(page_info.page_id);
        }

        let duration = start_time.elapsed();
        let completed_at = current_timestamp();

        let vacuum_stats = VacuumStats {
            pages_processed,
            rows_cleaned,
            space_reclaimed,
            duration_ms: duration.as_millis() as u64,
            vacuum_type,
            started_at,
            completed_at,
        };

        self.stats.write().insert(vacuum_type, vacuum_stats.clone());

        Ok(vacuum_stats)
    }

    async fn identify_vacuum_candidates(
        &self,
        buffer_pool: &Arc<BufferPool>,
        vacuum_type: VacuumType,
    ) -> Result<Vec<PageVacuumInfo>> {
        let mut candidates = Vec::new();
        let buffer_stats = buffer_pool.get_stats();

        for page_id in 1..=buffer_stats.total_pages as PageId {
            if let Ok(page_arc) = buffer_pool.get_page(page_id).await {
                let page = page_arc.read().await;
                let dead_row_info = self.analyze_page_deadness(&*page).await?;

                if self.should_vacuum_page(&dead_row_info, vacuum_type) {
                    let priority_score = self.calculate_vacuum_priority(&dead_row_info, page_id);

                    candidates.push(PageVacuumInfo {
                        page_id,
                        dead_rows: dead_row_info.dead_rows,
                        free_space: dead_row_info.free_space,
                        last_vacuum: self.get_last_vacuum_time(page_id),
                        priority_score,
                    });
                }
            }
        }

        candidates.sort_by_key(|info| std::cmp::Reverse(info.priority_score));

        Ok(match vacuum_type {
            VacuumType::Auto => candidates.into_iter().take(10).collect(),
            VacuumType::Manual | VacuumType::Analyze => candidates.into_iter().take(100).collect(),
            VacuumType::Full => candidates,
        })
    }

    async fn analyze_page_deadness(&self, page: &crate::page::Page) -> Result<DeadRowAnalysis> {
        let header = page.header();
        let mut dead_rows = 0;
        let mut live_rows = 0;
        let mut total_space_used = 0;

        for slot_id in 0..header.row_count {
            if let Some(row) = page.get_row(slot_id)? {
                if self.is_row_dead(&row) {
                    dead_rows += 1;
                } else {
                    live_rows += 1;
                }

                if let Some(slot) = page.get_slot(slot_id) {
                    total_space_used += slot.length as usize;
                }
            }
        }

        Ok(DeadRowAnalysis {
            dead_rows,
            live_rows,
            free_space: page.free_space(),
            total_space_used,
        })
    }

    fn is_row_dead(&self, row: &VectorRow) -> bool {
        row.deleted_xid.is_some()
    }

    fn should_vacuum_page(&self, analysis: &DeadRowAnalysis, vacuum_type: VacuumType) -> bool {
        let total_rows = analysis.dead_rows + analysis.live_rows;

        if total_rows == 0 {
            return false;
        }

        let dead_ratio = analysis.dead_rows as f32 / total_rows as f32;

        match vacuum_type {
            VacuumType::Auto => {
                dead_ratio > self.vacuum_threshold
                    && analysis.dead_rows > (self.vacuum_scale_factor * total_rows as f32) as usize
            }
            VacuumType::Manual => dead_ratio > 0.1,
            VacuumType::Analyze => true,
            VacuumType::Full => dead_ratio > 0.0,
        }
    }

    fn calculate_vacuum_priority(&self, analysis: &DeadRowAnalysis, page_id: PageId) -> u64 {
        let total_rows = analysis.dead_rows + analysis.live_rows;
        if total_rows == 0 {
            return 0;
        }

        let dead_ratio = analysis.dead_rows as f64 / total_rows as f64;
        let space_utilization = 1.0 - (analysis.free_space as f64 / crate::page::PAGE_SIZE as f64);

        let time_factor = if let Some(last_vacuum) = self.get_last_vacuum_time(page_id) {
            let hours_since = last_vacuum.elapsed().as_secs() / 3600;
            std::cmp::min(hours_since, 168) as f64 // Cap at 1 week
        } else {
            168.0 // Never vacuumed
        };

        let priority = (dead_ratio * 100.0) + (space_utilization * 50.0) + (time_factor * 2.0);
        priority as u64
    }

    async fn vacuum_page(
        &self,
        buffer_pool: &Arc<BufferPool>,
        page_id: PageId,
    ) -> Result<PageVacuumResult> {
        let page_arc = buffer_pool.get_page(page_id).await?;
        let mut page = page_arc.write().await;

        let initial_free_space = page.free_space();
        let mut rows_cleaned = 0;

        let mut live_rows = Vec::new();
        let header = page.header();

        for slot_id in 0..header.row_count {
            if let Some(row) = page.get_row(slot_id)? {
                if !self.is_row_dead(&row) {
                    live_rows.push(row);
                } else {
                    rows_cleaned += 1;
                }
            }
        }

        *page = crate::page::Page::new(page_id);

        for row in live_rows {
            page.insert_row(&row)?;
        }

        let final_free_space = page.free_space();
        let space_reclaimed = final_free_space - initial_free_space;

        Ok(PageVacuumResult {
            rows_cleaned: rows_cleaned as u64,
            space_reclaimed: space_reclaimed as u64,
        })
    }

    async fn should_throttle_vacuum(&self) -> bool {
        let current_cost = self.vacuum_cost_budget.load(Ordering::Relaxed);

        if current_cost >= self.vacuum_cost_limit {
            tokio::time::sleep(Duration::from_millis(10)).await;
            self.vacuum_cost_budget.store(0, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    async fn update_vacuum_cost(&self, cost: u64) {
        self.vacuum_cost_budget.fetch_add(cost, Ordering::Relaxed);
    }

    fn record_page_vacuum(&self, page_id: PageId) {
        self.last_vacuum_time
            .write()
            .insert(page_id, Instant::now());
    }

    fn get_last_vacuum_time(&self, page_id: PageId) -> Option<Instant> {
        self.last_vacuum_time.read().get(&page_id).copied()
    }

    pub async fn analyze_database(
        &self,
        buffer_pool: &Arc<BufferPool>,
        page_manager: &Arc<PageManager>,
    ) -> Result<DatabaseAnalysis> {
        let start_time = Instant::now();
        let buffer_stats = buffer_pool.get_stats();

        let mut total_pages = 0;
        let mut total_rows = 0;
        let mut total_dead_rows = 0;
        let mut total_free_space = 0;
        let mut page_utilization = Vec::new();

        for page_id in 1..=buffer_stats.total_pages as PageId {
            if let Ok(page_arc) = buffer_pool.get_page(page_id).await {
                let page = page_arc.read().await;
                let analysis = self.analyze_page_deadness(&*page).await?;

                total_pages += 1;
                total_rows += analysis.live_rows + analysis.dead_rows;
                total_dead_rows += analysis.dead_rows;
                total_free_space += analysis.free_space;

                let utilization =
                    1.0 - (analysis.free_space as f64 / crate::page::PAGE_SIZE as f64);
                page_utilization.push(utilization);
            }
        }

        let avg_utilization = if page_utilization.is_empty() {
            0.0
        } else {
            page_utilization.iter().sum::<f64>() / page_utilization.len() as f64
        };

        Ok(DatabaseAnalysis {
            total_pages,
            total_rows,
            total_dead_rows,
            total_free_space,
            average_page_utilization: avg_utilization,
            vacuum_recommendation: self.generate_vacuum_recommendation(total_rows, total_dead_rows),
            analysis_duration_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    fn generate_vacuum_recommendation(
        &self,
        total_rows: usize,
        dead_rows: usize,
    ) -> VacuumRecommendation {
        if total_rows == 0 {
            return VacuumRecommendation::NoAction;
        }

        let dead_ratio = dead_rows as f64 / total_rows as f64;

        if dead_ratio > 0.5 {
            VacuumRecommendation::FullVacuum
        } else if dead_ratio > 0.2 {
            VacuumRecommendation::StandardVacuum
        } else if dead_ratio > 0.05 {
            VacuumRecommendation::AutoVacuum
        } else {
            VacuumRecommendation::NoAction
        }
    }

    pub fn get_vacuum_stats(&self) -> HashMap<VacuumType, VacuumStats> {
        self.stats.read().clone()
    }

    pub fn set_auto_vacuum(&mut self, enabled: bool) {
        self.auto_vacuum_enabled = enabled;
    }

    pub fn set_vacuum_parameters(&mut self, threshold: f32, scale_factor: f32) {
        self.vacuum_threshold = threshold.clamp(0.01, 0.9);
        self.vacuum_scale_factor = scale_factor.clamp(0.01, 1.0);
    }

    pub async fn estimate_vacuum_time(&self, buffer_pool: &Arc<BufferPool>) -> Result<Duration> {
        let candidates = self
            .identify_vacuum_candidates(buffer_pool, VacuumType::Auto)
            .await?;
        let estimated_ms = candidates.len() as u64 * 50; // 50ms per page estimate
        Ok(Duration::from_millis(estimated_ms))
    }
}

#[derive(Debug)]
struct DeadRowAnalysis {
    dead_rows: usize,
    live_rows: usize,
    free_space: usize,
    total_space_used: usize,
}

#[derive(Debug)]
struct PageVacuumResult {
    rows_cleaned: u64,
    space_reclaimed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseAnalysis {
    pub total_pages: u64,
    pub total_rows: usize,
    pub total_dead_rows: usize,
    pub total_free_space: usize,
    pub average_page_utilization: f64,
    pub vacuum_recommendation: VacuumRecommendation,
    pub analysis_duration_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VacuumRecommendation {
    NoAction,
    AutoVacuum,
    StandardVacuum,
    FullVacuum,
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
