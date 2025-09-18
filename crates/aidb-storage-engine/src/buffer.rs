use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::RwLock as AsyncRwLock;

use crate::page::{Page, PageManager};
use crate::{PageId, Result};

#[derive(Debug, Clone)]
pub struct BufferStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub pages_written: u64,
    pub total_pages: usize,
    pub dirty_pages: usize,
}

#[derive(Debug)]
struct BufferFrame {
    page: Arc<AsyncRwLock<Page>>,
    usage_count: AtomicUsize,
    last_accessed: AtomicU64,
    access_frequency: AtomicUsize,
}

impl BufferFrame {
    fn new(page: Page) -> Self {
        Self {
            page: Arc::new(AsyncRwLock::new(page)),
            usage_count: AtomicUsize::new(1),
            last_accessed: AtomicU64::new(current_timestamp()),
            access_frequency: AtomicUsize::new(1),
        }
    }

    fn access(&self) {
        self.usage_count.fetch_add(1, Ordering::Relaxed);
        self.access_frequency.fetch_add(1, Ordering::Relaxed);
        self.last_accessed
            .store(current_timestamp(), Ordering::Relaxed);
    }

    fn get_priority_score(&self) -> u64 {
        let usage = self.usage_count.load(Ordering::Relaxed) as u64;
        let frequency = self.access_frequency.load(Ordering::Relaxed) as u64;
        let recency = current_timestamp() - self.last_accessed.load(Ordering::Relaxed);

        (usage * frequency * 1000) / (recency + 1)
    }
}

pub struct BufferPool {
    frames: DashMap<PageId, BufferFrame>,
    page_manager: Option<Arc<PageManager>>,
    max_pages: usize,
    current_size: AtomicUsize,
    stats: Arc<RwLock<BufferStats>>,
    clock_hand: AtomicUsize,
}

impl BufferPool {
    pub async fn new(max_pages: usize) -> Result<Self> {
        Ok(Self {
            frames: DashMap::new(),
            page_manager: None,
            max_pages,
            current_size: AtomicUsize::new(0),
            stats: Arc::new(RwLock::new(BufferStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                pages_written: 0,
                total_pages: 0,
                dirty_pages: 0,
            })),
            clock_hand: AtomicUsize::new(0),
        })
    }

    pub fn set_page_manager(&mut self, page_manager: Arc<PageManager>) {
        self.page_manager = Some(page_manager);
    }

    pub async fn get_page(&self, page_id: PageId) -> Result<Arc<AsyncRwLock<Page>>> {
        if let Some(frame) = self.frames.get(&page_id) {
            frame.access();
            self.stats.write().hits += 1;
            return Ok(frame.page.clone());
        }

        self.load_page(page_id).await
    }

    async fn load_page(&self, page_id: PageId) -> Result<Arc<AsyncRwLock<Page>>> {
        let page_manager = self.page_manager.as_ref().ok_or_else(|| {
            crate::StorageEngineError::PageCorruption("No page manager".to_string())
        })?;

        let page = page_manager.read_page(page_id).await?;

        if self.current_size.load(Ordering::Relaxed) >= self.max_pages {
            self.evict_pages().await?;
        }

        let frame = BufferFrame::new(page);
        let page_ref = frame.page.clone();

        self.frames.insert(page_id, frame);
        self.current_size.fetch_add(1, Ordering::Relaxed);

        let mut stats = self.stats.write();
        stats.misses += 1;
        stats.total_pages = self.current_size.load(Ordering::Relaxed);
        drop(stats);

        Ok(page_ref)
    }

    async fn evict_pages(&self) -> Result<()> {
        let target_evictions = self.max_pages / 4; // Evict 25% when full
        let mut evicted = 0;

        // Use adaptive replacement algorithm (ARC-inspired)
        let mut candidates: Vec<(PageId, u64)> = self
            .frames
            .iter()
            .map(|entry| {
                let page_id = *entry.key();
                let priority = entry.value().get_priority_score();
                (page_id, priority)
            })
            .collect();

        candidates.sort_by_key(|&(_, priority)| priority);

        // Simplified eviction without complex borrow issues
        let evict_candidates: Vec<_> = candidates.into_iter().take(target_evictions).collect();
        for (page_id, _) in evict_candidates {
            if self.frames.contains_key(&page_id) {
                evicted += 1;
                self.frames.remove(&page_id);
                self.current_size.fetch_sub(1, Ordering::Relaxed);
            }
        }

        let mut stats = self.stats.write();
        stats.evictions += evicted as u64;
        stats.total_pages = self.current_size.load(Ordering::Relaxed);

        Ok(())
    }

    pub async fn flush_all(&self) -> Result<()> {
        let page_manager = self.page_manager.as_ref().ok_or_else(|| {
            crate::StorageEngineError::PageCorruption("No page manager".to_string())
        })?;

        let mut pages_written = 0;

        for entry in self.frames.iter() {
            let mut page_guard = entry.value().page.write().await;
            if page_guard.dirty {
                page_manager.write_page(&mut *page_guard).await?;
                pages_written += 1;
            }
        }

        self.stats.write().pages_written += pages_written;
        Ok(())
    }

    pub async fn flush_page(&self, page_id: PageId) -> Result<bool> {
        if let Some(frame) = self.frames.get(&page_id) {
            let mut page_guard = frame.page.write().await;
            if page_guard.dirty {
                if let Some(page_manager) = &self.page_manager {
                    page_manager.write_page(&mut *page_guard).await?;
                    self.stats.write().pages_written += 1;
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    pub fn get_stats(&self) -> BufferStats {
        let stats = self.stats.read();
        BufferStats {
            hits: stats.hits,
            misses: stats.misses,
            evictions: stats.evictions,
            pages_written: stats.pages_written,
            total_pages: self.current_size.load(Ordering::Relaxed),
            dirty_pages: self.count_dirty_pages(),
        }
    }

    fn count_dirty_pages(&self) -> usize {
        self.frames
            .iter()
            .filter(|entry| {
                if let Ok(page_guard) = entry.value().page.try_read() {
                    page_guard.dirty
                } else {
                    false
                }
            })
            .count()
    }

    pub async fn prefetch_pages(&self, page_ids: &[PageId]) -> Result<()> {
        for &page_id in page_ids {
            if !self.frames.contains_key(&page_id) {
                // Note: Simplified prefetch - would need Arc cloning for async spawning
                let _ = self.get_page(page_id).await?;
            }
        }
        Ok(())
    }

    pub fn get_memory_usage(&self) -> usize {
        self.current_size.load(Ordering::Relaxed) * crate::page::PAGE_SIZE
    }

    pub fn get_hit_ratio(&self) -> f64 {
        let stats = self.stats.read();
        let total = stats.hits + stats.misses;
        if total == 0 {
            0.0
        } else {
            stats.hits as f64 / total as f64
        }
    }

    pub async fn warm_up(&self, frequently_accessed_pages: &[PageId]) -> Result<()> {
        for &page_id in frequently_accessed_pages {
            let _ = self.get_page(page_id).await?;
        }
        Ok(())
    }

    pub async fn checkpoint(&self) -> Result<()> {
        let page_manager = self.page_manager.as_ref().ok_or_else(|| {
            crate::StorageEngineError::PageCorruption("No page manager".to_string())
        })?;

        let mut dirty_pages = Vec::new();

        for entry in self.frames.iter() {
            let page_guard = entry.value().page.read().await;
            if page_guard.dirty {
                dirty_pages.push((entry.key().clone(), entry.value().page.clone()));
            }
        }

        let mut pages_written = 0;
        for (page_id, page_arc) in dirty_pages {
            let mut page_guard = page_arc.write().await;
            page_manager.write_page(&mut *page_guard).await?;
            pages_written += 1;
        }

        self.stats.write().pages_written += pages_written;
        Ok(())
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
