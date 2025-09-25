use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use thiserror::Error;

use super::compaction::CompactionStrategy;
use super::memtable::MemTable;
use super::scheduler::{CompactionExecutor, CompactionScheduler, CompactionTask, SchedulerError};
use super::sstable::{merge_tables, SSTable};
use super::wal::WriteAheadLog;

#[derive(Clone)]
pub struct LSMTreeConfig {
    pub data_dir: PathBuf,
    pub memtable_capacity_bytes: usize,
    pub sstable_block_size: usize,
    pub bloom_bits_per_key: usize,
    pub bloom_hashes: u32,
    pub compaction_strategy: CompactionStrategy,
    pub max_levels: usize,
}

impl Default for LSMTreeConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data/lsm"),
            memtable_capacity_bytes: 4 * 1024 * 1024,
            sstable_block_size: 128,
            bloom_bits_per_key: 10,
            bloom_hashes: 6,
            compaction_strategy: CompactionStrategy::default(),
            max_levels: 4,
        }
    }
}

#[derive(Error, Debug)]
pub enum LsmError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("scheduler error: {0}")]
    Scheduler(#[from] SchedulerError),
}

impl From<bincode::Error> for LsmError {
    fn from(value: bincode::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

pub type LsmResult<T> = Result<T, LsmError>;

pub struct LSMTree {
    inner: Arc<LsmInner>,
    scheduler: CompactionScheduler,
}

impl LSMTree {
    pub async fn open(config: LSMTreeConfig) -> LsmResult<Self> {
        fs::create_dir_all(&config.data_dir)?;
        let wal_dir = config.data_dir.join("wal");
        let sstable_dir = config.data_dir.join("sstables");
        fs::create_dir_all(&wal_dir)?;
        fs::create_dir_all(&sstable_dir)?;
        let inner = Arc::new(LsmInner::new(config, wal_dir, sstable_dir)?);
        let executor: Arc<dyn CompactionExecutor> = inner.clone() as Arc<dyn CompactionExecutor>;
        let scheduler = CompactionScheduler::new(executor);
        Ok(Self { inner, scheduler })
    }

    pub async fn put(&self, key: Vec<u8>, value: Vec<u8>) -> LsmResult<()> {
        let memtable = self.inner.active_memtable();
        memtable.insert(key, value)?;
        if let Some(to_flush) = self.inner.prepare_memtable_rotation(&memtable)? {
            self.scheduler.schedule_flush(to_flush).await?;
        }
        Ok(())
    }

    pub async fn delete(&self, key: Vec<u8>) -> LsmResult<()> {
        let memtable = self.inner.active_memtable();
        memtable.delete(key)?;
        if let Some(to_flush) = self.inner.prepare_memtable_rotation(&memtable)? {
            self.scheduler.schedule_flush(to_flush).await?;
        }
        Ok(())
    }

    pub async fn get(&self, key: &[u8]) -> LsmResult<Option<Vec<u8>>> {
        if let Some(result) = self.inner.lookup_memtables(key) {
            return Ok(result);
        }
        self.inner.lookup_sstables(key)
    }

    pub fn stats(&self) -> LsmStats {
        self.inner.stats()
    }
}

pub struct LsmStats {
    pub active_memtable_entries: usize,
    pub immutable_memtables: usize,
    pub level_counts: Vec<usize>,
}

struct LsmState {
    active_memtable: Arc<MemTable>,
    immutable_memtables: VecDeque<Arc<MemTable>>,
    levels: Vec<Vec<Arc<SSTable>>>,
}

struct LsmInner {
    config: LSMTreeConfig,
    wal_dir: PathBuf,
    sstable_dir: PathBuf,
    state: RwLock<LsmState>,
    next_memtable_id: AtomicU64,
    next_sstable_id: AtomicU64,
}

impl LsmInner {
    fn new(config: LSMTreeConfig, wal_dir: PathBuf, sstable_dir: PathBuf) -> LsmResult<Self> {
        let mut next_sstable_id = 1u64;
        let mut level0_tables = Vec::new();
        for entry in fs::read_dir(&sstable_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let path = entry.path();
            let file_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            if let Some(id_str) = file_name.strip_prefix("sstable-") {
                if let Ok(id) = id_str.parse::<u64>() {
                    let table = SSTable::open(&path, id)?;
                    next_sstable_id = next_sstable_id.max(id + 1);
                    level0_tables.push(Arc::new(table));
                }
            }
        }

        let next_memtable_id = AtomicU64::new(1);
        let active_memtable = Self::create_memtable_internal(&config, &wal_dir, &next_memtable_id)?;

        let mut levels = vec![Vec::new(); config.max_levels.max(1)];
        level0_tables.sort_by_key(|table| table.id());
        levels[0] = level0_tables;

        Ok(Self {
            config,
            wal_dir,
            sstable_dir,
            state: RwLock::new(LsmState {
                active_memtable,
                immutable_memtables: VecDeque::new(),
                levels,
            }),
            next_memtable_id,
            next_sstable_id: AtomicU64::new(next_sstable_id),
        })
    }

    fn create_memtable_internal(
        config: &LSMTreeConfig,
        wal_dir: &Path,
        next_memtable_id: &AtomicU64,
    ) -> LsmResult<Arc<MemTable>> {
        let id = next_memtable_id.fetch_add(1, Ordering::SeqCst);
        let wal_path = wal_dir.join(format!("memtable-{id}.wal"));
        let wal = Arc::new(WriteAheadLog::open(wal_path)?);
        let memtable = MemTable::new(id, wal, config.memtable_capacity_bytes)?;
        Ok(Arc::new(memtable))
    }

    fn active_memtable(&self) -> Arc<MemTable> {
        self.state.read().active_memtable.clone()
    }

    fn prepare_memtable_rotation(
        &self,
        expected: &Arc<MemTable>,
    ) -> LsmResult<Option<Arc<MemTable>>> {
        if !expected.is_full() {
            return Ok(None);
        }
        let mut state = self.state.write();
        if state.active_memtable.id() != expected.id() || !state.active_memtable.is_full() {
            return Ok(None);
        }
        let flush_candidate = state.active_memtable.clone();
        let new_memtable =
            Self::create_memtable_internal(&self.config, &self.wal_dir, &self.next_memtable_id)?;
        state.active_memtable = new_memtable;
        state.immutable_memtables.push_back(flush_candidate.clone());
        Ok(Some(flush_candidate))
    }

    fn lookup_memtables(&self, key: &[u8]) -> Option<Option<Vec<u8>>> {
        let state = self.state.read();
        if let Some(result) = state.active_memtable.get(key) {
            return Some(result);
        }
        for mem in state.immutable_memtables.iter().rev() {
            if let Some(result) = mem.get(key) {
                return Some(result);
            }
        }
        None
    }

    fn lookup_sstables(&self, key: &[u8]) -> LsmResult<Option<Vec<u8>>> {
        let state = self.state.read();
        for level in &state.levels {
            for table in level.iter().rev() {
                if let Some(result) = table.get(key)? {
                    return Ok(result);
                }
            }
        }
        Ok(None)
    }

    fn stats(&self) -> LsmStats {
        let state = self.state.read();
        LsmStats {
            active_memtable_entries: state.active_memtable.len(),
            immutable_memtables: state.immutable_memtables.len(),
            level_counts: state.levels.iter().map(|level| level.len()).collect(),
        }
    }

    fn finalize_memtable(
        &self,
        memtable_id: u64,
        table: Option<Arc<SSTable>>,
        wal_path: &Path,
    ) -> LsmResult<Vec<CompactionTask>> {
        {
            let mut state = self.state.write();
            state
                .immutable_memtables
                .retain(|mem| mem.id() != memtable_id);
            if state.levels.is_empty() {
                state
                    .levels
                    .resize(self.config.max_levels.max(1), Vec::new());
            }
            if let Some(table) = table {
                state.levels[0].push(table);
            }
        }
        let _ = fs::remove_file(wal_path);
        Ok(self.evaluate_compaction())
    }

    fn evaluate_compaction(&self) -> Vec<CompactionTask> {
        let state = self.state.read();
        let level_sizes: Vec<usize> = state.levels.iter().map(|level| level.len()).collect();
        let mut tasks = Vec::new();
        for level in 0..level_sizes.len() {
            if self
                .config
                .compaction_strategy
                .should_compact(level, &level_sizes)
            {
                tasks.push(CompactionTask::CompactLevel(level));
            }
        }
        tasks
    }

    fn perform_compaction(&self, level: usize) -> LsmResult<Option<(Arc<SSTable>, usize)>> {
        let (tables, next_level) = {
            let mut state = self.state.write();
            if level >= state.levels.len() {
                return Ok(None);
            }
            let level_sizes: Vec<usize> = state.levels.iter().map(|lvl| lvl.len()).collect();
            let target = self
                .config
                .compaction_strategy
                .tables_to_compact(level, &level_sizes);
            if target == 0 || state.levels[level].is_empty() {
                return Ok(None);
            }
            let count = target.min(state.levels[level].len());
            let drained: Vec<Arc<SSTable>> = state.levels[level].drain(..count).collect();
            let next_level = (level + 1).min(self.config.max_levels.saturating_sub(1));
            (drained, next_level)
        };

        if tables.is_empty() {
            return Ok(None);
        }

        let new_id = self.next_sstable_id.fetch_add(1, Ordering::SeqCst);
        let new_path = self.sstable_dir.join(format!("sstable-{new_id}.sst"));
        let merged = merge_tables(
            &new_path,
            new_id,
            &tables,
            self.config.sstable_block_size,
            self.config.bloom_bits_per_key,
            self.config.bloom_hashes,
        )?;
        for table in &tables {
            let _ = fs::remove_file(table.path());
        }
        Ok(Some((Arc::new(merged), next_level)))
    }
}

#[async_trait::async_trait]
impl CompactionExecutor for LsmInner {
    async fn flush_memtable(&self, memtable: Arc<MemTable>) -> LsmResult<Vec<CompactionTask>> {
        let entries = memtable.snapshot();
        let wal_path = memtable.wal_path().to_path_buf();
        if entries.is_empty() {
            return self.finalize_memtable(memtable.id(), None, &wal_path);
        }
        let table_id = self.next_sstable_id.fetch_add(1, Ordering::SeqCst);
        let table_path = self.sstable_dir.join(format!("sstable-{table_id}.sst"));
        let table = SSTable::create(
            &table_path,
            table_id,
            &entries,
            self.config.sstable_block_size,
            self.config.bloom_bits_per_key,
            self.config.bloom_hashes,
        )?;
        self.finalize_memtable(memtable.id(), Some(Arc::new(table)), &wal_path)
    }

    async fn compact_level(&self, level: usize) -> LsmResult<Vec<CompactionTask>> {
        if let Some((merged, next_level)) = self.perform_compaction(level)? {
            {
                let mut state = self.state.write();
                if state.levels.len() <= next_level {
                    state.levels.resize(next_level + 1, Vec::new());
                }
                state.levels[next_level].push(merged);
            }
        }
        Ok(self.evaluate_compaction())
    }
}
