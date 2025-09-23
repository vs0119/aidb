use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{Result, StorageEngineError, VectorRow};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    LZ4,
    Zstd,
    Delta,
    Dictionary,
}

#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
    pub compression_time_ns: u64,
    pub decompression_time_ns: u64,
    pub algorithm_used: CompressionAlgorithm,
}

pub struct CompressionManager {
    stats: RwLock<HashMap<CompressionAlgorithm, CompressionStats>>,
    adaptive_threshold: AtomicU64,
    dictionary_cache: RwLock<HashMap<String, Vec<u8>>>,
}

impl CompressionManager {
    pub fn new() -> Self {
        Self {
            stats: RwLock::new(HashMap::new()),
            adaptive_threshold: AtomicU64::new(1024), // Switch algorithms for data > 1KB
            dictionary_cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn compress(&self, data: &[u8]) -> Result<CompressedData> {
        let algorithm = self.select_algorithm(data.len());
        let start = std::time::Instant::now();

        let compressed = match algorithm {
            CompressionAlgorithm::None => data.to_vec(),
            CompressionAlgorithm::LZ4 => self.compress_lz4(data)?,
            CompressionAlgorithm::Zstd => self.compress_zstd(data)?,
            CompressionAlgorithm::Delta => self.compress_delta(data)?,
            CompressionAlgorithm::Dictionary => self.compress_dictionary(data)?,
        };

        let compression_time = start.elapsed().as_nanos() as u64;

        let stats = CompressionStats {
            original_size: data.len() as u64,
            compressed_size: compressed.len() as u64,
            compression_ratio: data.len() as f64 / compressed.len() as f64,
            compression_time_ns: compression_time,
            decompression_time_ns: 0,
            algorithm_used: algorithm,
        };

        self.update_stats(algorithm, stats);

        Ok(CompressedData {
            algorithm,
            data: compressed,
            original_size: data.len(),
        })
    }

    pub fn decompress(&self, compressed: &CompressedData) -> Result<Vec<u8>> {
        let start = std::time::Instant::now();

        let decompressed = match compressed.algorithm {
            CompressionAlgorithm::None => compressed.data.clone(),
            CompressionAlgorithm::LZ4 => self.decompress_lz4(&compressed.data)?,
            CompressionAlgorithm::Zstd => self.decompress_zstd(&compressed.data)?,
            CompressionAlgorithm::Delta => self.decompress_delta(&compressed.data)?,
            CompressionAlgorithm::Dictionary => self.decompress_dictionary(&compressed.data)?,
        };

        let decompression_time = start.elapsed().as_nanos() as u64;

        if let Some(stats) = self.stats.write().get_mut(&compressed.algorithm) {
            stats.decompression_time_ns = decompression_time;
        }

        Ok(decompressed)
    }

    fn select_algorithm(&self, data_size: usize) -> CompressionAlgorithm {
        if data_size < 64 {
            return CompressionAlgorithm::None;
        }

        let threshold = self.adaptive_threshold.load(Ordering::Relaxed) as usize;

        if data_size < threshold {
            CompressionAlgorithm::LZ4
        } else {
            let stats = self.stats.read();

            let lz4_ratio = stats
                .get(&CompressionAlgorithm::LZ4)
                .map(|s| s.compression_ratio)
                .unwrap_or(1.0);

            let zstd_ratio = stats
                .get(&CompressionAlgorithm::Zstd)
                .map(|s| s.compression_ratio)
                .unwrap_or(1.0);

            if zstd_ratio > lz4_ratio * 1.2 {
                CompressionAlgorithm::Zstd
            } else {
                CompressionAlgorithm::LZ4
            }
        }
    }

    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>> {
        lz4_flex::decompress_size_prepended(data).map_err(|e| {
            StorageEngineError::CompressionError(format!("LZ4 decompression failed: {}", e))
        })
    }

    fn compress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::encode_all(data, 3).map_err(|e| {
            StorageEngineError::CompressionError(format!("Zstd compression failed: {}", e))
        })
    }

    fn decompress_zstd(&self, data: &[u8]) -> Result<Vec<u8>> {
        zstd::decode_all(data).map_err(|e| {
            StorageEngineError::CompressionError(format!("Zstd decompression failed: {}", e))
        })
    }

    fn compress_delta(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 4 {
            return Ok(data.to_vec());
        }

        let mut compressed = Vec::with_capacity(data.len());
        compressed.push(data[0]);

        for i in 1..data.len() {
            let delta = data[i].wrapping_sub(data[i - 1]);
            compressed.push(delta);
        }

        self.compress_lz4(&compressed)
    }

    fn decompress_delta(&self, data: &[u8]) -> Result<Vec<u8>> {
        let lz4_decompressed = self.decompress_lz4(data)?;

        if lz4_decompressed.is_empty() {
            return Ok(Vec::new());
        }

        let mut decompressed = Vec::with_capacity(lz4_decompressed.len());
        decompressed.push(lz4_decompressed[0]);

        for i in 1..lz4_decompressed.len() {
            let prev = decompressed[i - 1];
            let current = prev.wrapping_add(lz4_decompressed[i]);
            decompressed.push(current);
        }

        Ok(decompressed)
    }

    fn compress_dictionary(&self, data: &[u8]) -> Result<Vec<u8>> {
        let dict_key = self.generate_dictionary_key(data);

        if let Some(_dictionary) = self.dictionary_cache.read().get(&dict_key).cloned() {
            let compressed = zstd::bulk::compress(data, 3).map_err(|e| {
                StorageEngineError::CompressionError(format!(
                    "Dictionary compression failed: {}",
                    e
                ))
            })?;

            let mut result = Vec::with_capacity(dict_key.len() + 4 + compressed.len());
            result.extend_from_slice(&(dict_key.len() as u32).to_le_bytes());
            result.extend_from_slice(dict_key.as_bytes());
            result.extend_from_slice(&compressed);

            return Ok(result);
        }

        self.compress_zstd(data)
    }

    fn decompress_dictionary(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() < 4 {
            return self.decompress_zstd(data);
        }

        let dict_key_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if data.len() < 4 + dict_key_len {
            return self.decompress_zstd(data);
        }

        let dict_key = String::from_utf8_lossy(&data[4..4 + dict_key_len]).to_string();
        let compressed_data = &data[4 + dict_key_len..];

        if let Some(_dictionary) = self.dictionary_cache.read().get(&dict_key).cloned() {
            zstd::bulk::decompress(compressed_data, 1024 * 1024).map_err(|e| {
                StorageEngineError::CompressionError(format!(
                    "Dictionary decompression failed: {}",
                    e
                ))
            })
        } else {
            self.decompress_zstd(compressed_data)
        }
    }

    fn generate_dictionary_key(&self, data: &[u8]) -> String {
        if data.len() < 16 {
            return "small".to_string();
        }

        let sample_size = std::cmp::min(256, data.len());
        let sample = &data[..sample_size];

        let mut histogram = [0u32; 256];
        for &byte in sample {
            histogram[byte as usize] += 1;
        }

        let mut key_components = Vec::new();
        for (i, &count) in histogram.iter().enumerate() {
            if count > sample_size as u32 / 20 {
                key_components.push(format!("{}:{}", i, count));
            }
        }

        if key_components.is_empty() {
            "random".to_string()
        } else {
            key_components.join("-")
        }
    }

    fn update_stats(&self, algorithm: CompressionAlgorithm, stats: CompressionStats) {
        let mut stats_map = self.stats.write();

        if let Some(existing) = stats_map.get_mut(&algorithm) {
            existing.original_size += stats.original_size;
            existing.compressed_size += stats.compressed_size;
            existing.compression_ratio =
                existing.original_size as f64 / existing.compressed_size as f64;
            existing.compression_time_ns =
                (existing.compression_time_ns + stats.compression_time_ns) / 2;
        } else {
            stats_map.insert(algorithm, stats);
        }

        self.adapt_threshold(&stats_map);
    }

    fn adapt_threshold(&self, stats_map: &HashMap<CompressionAlgorithm, CompressionStats>) {
        if let (Some(lz4_stats), Some(zstd_stats)) = (
            stats_map.get(&CompressionAlgorithm::LZ4),
            stats_map.get(&CompressionAlgorithm::Zstd),
        ) {
            let lz4_speed = lz4_stats.original_size as f64 / lz4_stats.compression_time_ns as f64;
            let zstd_speed =
                zstd_stats.original_size as f64 / zstd_stats.compression_time_ns as f64;

            if lz4_speed > zstd_speed * 2.0 {
                let current_threshold = self.adaptive_threshold.load(Ordering::Relaxed);
                let new_threshold = std::cmp::min(current_threshold + 512, 8192);
                self.adaptive_threshold
                    .store(new_threshold, Ordering::Relaxed);
            } else if zstd_stats.compression_ratio > lz4_stats.compression_ratio * 1.5 {
                let current_threshold = self.adaptive_threshold.load(Ordering::Relaxed);
                let new_threshold = std::cmp::max(current_threshold.saturating_sub(256), 256);
                self.adaptive_threshold
                    .store(new_threshold, Ordering::Relaxed);
            }
        }
    }

    pub fn get_stats(&self) -> HashMap<CompressionAlgorithm, CompressionStats> {
        self.stats.read().clone()
    }

    pub fn get_overall_compression_ratio(&self) -> f64 {
        let stats = self.stats.read();
        let total_original: u64 = stats.values().map(|s| s.original_size).sum();
        let total_compressed: u64 = stats.values().map(|s| s.compressed_size).sum();

        if total_compressed == 0 {
            1.0
        } else {
            total_original as f64 / total_compressed as f64
        }
    }

    pub fn train_dictionary(&self, training_data: &[&[u8]], key: String) -> Result<()> {
        if training_data.len() < 10 {
            return Ok(());
        }

        let combined: Vec<u8> = training_data
            .iter()
            .flat_map(|data| data.iter())
            .copied()
            .collect();

        if combined.len() < 1024 {
            return Ok(());
        }

        let sample_size = std::cmp::min(combined.len(), 100 * 1024);
        let sample = &combined[..sample_size];

        let dictionary = sample.to_vec();

        self.dictionary_cache.write().insert(key, dictionary);
        Ok(())
    }

    pub fn compress_vector_row(&self, row: &VectorRow) -> Result<CompressedVectorRow> {
        let serialized = bincode::serialize(row).map_err(|e| {
            StorageEngineError::CompressionError(format!("Row serialization failed: {}", e))
        })?;

        let compressed_data = self.compress(&serialized)?;

        Ok(CompressedVectorRow {
            id: row.id,
            compressed_data,
            created_xid: row.created_xid,
            updated_xid: row.updated_xid,
            deleted_xid: row.deleted_xid,
        })
    }

    pub fn decompress_vector_row(&self, compressed_row: &CompressedVectorRow) -> Result<VectorRow> {
        let decompressed_data = self.decompress(&compressed_row.compressed_data)?;

        let mut row: VectorRow = bincode::deserialize(&decompressed_data).map_err(|e| {
            StorageEngineError::CompressionError(format!("Row deserialization failed: {}", e))
        })?;

        row.created_xid = compressed_row.created_xid;
        row.updated_xid = compressed_row.updated_xid;
        row.deleted_xid = compressed_row.deleted_xid;

        Ok(row)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompressedData {
    pub algorithm: CompressionAlgorithm,
    pub data: Vec<u8>,
    pub original_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVectorRow {
    pub id: uuid::Uuid,
    pub compressed_data: CompressedData,
    pub created_xid: u64,
    pub updated_xid: Option<u64>,
    pub deleted_xid: Option<u64>,
}
