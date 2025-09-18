use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// Adaptive index selection based on workload characteristics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkloadStats {
    pub query_latencies: VecDeque<f64>,
    pub ingest_rates: VecDeque<f64>,
    pub memory_usage: VecDeque<usize>,
    pub query_patterns: QueryPatternAnalyzer,
    pub dataset_characteristics: DatasetCharacteristics,
    pub window_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryPatternAnalyzer {
    pub top_k_distribution: Vec<usize>, // Histogram of top_k values
    pub filter_usage: f64,              // Percentage of queries with filters
    pub dimension_access_patterns: Vec<f64>, // Which dimensions are queried most
    pub query_frequency: f64,           // Queries per second
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetCharacteristics {
    pub size: usize,
    pub dimensionality: usize,
    pub intrinsic_dimensionality: f64, // Estimated via PCA or other methods
    pub clustering_coefficient: f64,   // How clustered the data is
    pub update_frequency: f64,         // Updates per second
    pub memory_constraints: usize,     // Available memory in bytes
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OptimalIndex {
    BruteForce,
    HNSW {
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    },
    ProductQuantizedHNSW {
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        pq_m: usize,
    },
    HybridIndex {
        primary: Box<OptimalIndex>,
        fallback: Box<OptimalIndex>,
    },
}

impl WorkloadStats {
    pub fn new(window_size: usize) -> Self {
        Self {
            query_latencies: VecDeque::with_capacity(window_size),
            ingest_rates: VecDeque::with_capacity(window_size),
            memory_usage: VecDeque::with_capacity(window_size),
            query_patterns: QueryPatternAnalyzer::new(),
            dataset_characteristics: DatasetCharacteristics::default(),
            window_size,
        }
    }

    pub fn record_query(&mut self, latency: f64, top_k: usize, has_filter: bool) {
        if self.query_latencies.len() >= self.window_size {
            self.query_latencies.pop_front();
        }
        self.query_latencies.push_back(latency);

        self.query_patterns.record_query(top_k, has_filter);
    }

    pub fn record_ingest(&mut self, rate: f64, memory_usage: usize) {
        if self.ingest_rates.len() >= self.window_size {
            self.ingest_rates.pop_front();
            self.memory_usage.pop_front();
        }
        self.ingest_rates.push_back(rate);
        self.memory_usage.push_back(memory_usage);
    }

    // Analyze current workload and recommend optimal index
    pub fn recommend_index(&self) -> OptimalIndex {
        let avg_latency = self.avg_latency();
        let p95_latency = self.p95_latency();
        let memory_pressure = self.memory_pressure();
        let query_freq = self.query_patterns.query_frequency;
        let dataset_size = self.dataset_characteristics.size;

        // Decision tree for index selection
        if dataset_size < 10_000 {
            // Small datasets: use brute force for simplicity and accuracy
            OptimalIndex::BruteForce
        } else if memory_pressure > 0.8 && dataset_size > 100_000 {
            // High memory pressure: use Product Quantization
            OptimalIndex::ProductQuantizedHNSW {
                m: self.optimal_hnsw_m(),
                ef_construction: self.optimal_ef_construction(),
                ef_search: self.optimal_ef_search(avg_latency, p95_latency),
                pq_m: self.optimal_pq_m(),
            }
        } else if query_freq > 1000.0 && p95_latency > 10.0 {
            // High QPS with latency concerns: optimized HNSW
            OptimalIndex::HNSW {
                m: self.optimal_hnsw_m(),
                ef_construction: self.optimal_ef_construction(),
                ef_search: self.optimal_ef_search(avg_latency, p95_latency),
            }
        } else if self.dataset_characteristics.update_frequency > 100.0 {
            // High update frequency: hybrid approach
            OptimalIndex::HybridIndex {
                primary: Box::new(OptimalIndex::HNSW {
                    m: 8, // Lower m for faster updates
                    ef_construction: 100,
                    ef_search: 32,
                }),
                fallback: Box::new(OptimalIndex::BruteForce),
            }
        } else {
            // Default: balanced HNSW
            OptimalIndex::HNSW {
                m: 16,
                ef_construction: 200,
                ef_search: 50,
            }
        }
    }

    pub fn avg_latency(&self) -> f64 {
        if self.query_latencies.is_empty() {
            return 0.0;
        }
        self.query_latencies.iter().sum::<f64>() / self.query_latencies.len() as f64
    }

    fn p95_latency(&self) -> f64 {
        if self.query_latencies.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.query_latencies.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (0.95 * sorted.len() as f64) as usize;
        sorted.get(index).copied().unwrap_or(0.0)
    }

    pub fn memory_pressure(&self) -> f64 {
        if self.memory_usage.is_empty() {
            return 0.0;
        }
        let current_usage = *self.memory_usage.back().unwrap() as f64;
        let available = self.dataset_characteristics.memory_constraints as f64;
        if available > 0.0 {
            current_usage / available
        } else {
            0.0
        }
    }

    fn optimal_hnsw_m(&self) -> usize {
        // Higher m for better accuracy, lower for memory efficiency
        let memory_factor = 1.0 - self.memory_pressure();
        let base_m = 16;
        ((base_m as f64 * (0.5 + memory_factor)) as usize).clamp(4, 64)
    }

    fn optimal_ef_construction(&self) -> usize {
        // Higher ef_construction for better index quality
        let dataset_factor = (self.dataset_characteristics.size as f64).log10() / 6.0; // Scale with dataset size
        let base_ef = 200;
        ((base_ef as f64 * (0.5 + dataset_factor)) as usize).clamp(50, 1000)
    }

    fn optimal_ef_search(&self, _avg_latency: f64, p95_latency: f64) -> usize {
        // Lower ef_search for lower latency, higher for better accuracy
        let latency_factor = if p95_latency > 5.0 { 0.5 } else { 1.0 }; // Reduce if latency is high
        let top_k_factor = self.query_patterns.avg_top_k() as f64 / 10.0; // Scale with typical top_k
        let base_ef = 50;
        ((base_ef as f64 * latency_factor * (1.0 + top_k_factor)) as usize).clamp(10, 500)
    }

    fn optimal_pq_m(&self) -> usize {
        // Number of subvectors for Product Quantization
        let dim = self.dataset_characteristics.dimensionality;
        if dim <= 128 {
            8
        } else if dim <= 512 {
            16
        } else {
            32
        }
    }
}

impl QueryPatternAnalyzer {
    pub fn new() -> Self {
        Self {
            top_k_distribution: vec![0; 101], // Support top_k up to 100
            filter_usage: 0.0,
            dimension_access_patterns: Vec::new(),
            query_frequency: 0.0,
        }
    }

    pub fn record_query(&mut self, top_k: usize, has_filter: bool) {
        if top_k <= 100 {
            self.top_k_distribution[top_k] += 1;
        }

        // Update filter usage (exponential moving average)
        let filter_indicator = if has_filter { 1.0 } else { 0.0 };
        self.filter_usage = 0.9 * self.filter_usage + 0.1 * filter_indicator;
    }

    pub fn avg_top_k(&self) -> usize {
        let total_queries: usize = self.top_k_distribution.iter().sum();
        if total_queries == 0 {
            return 10; // Default
        }

        let weighted_sum: usize = self
            .top_k_distribution
            .iter()
            .enumerate()
            .map(|(k, count)| k * count)
            .sum();

        weighted_sum / total_queries
    }
}

impl Default for DatasetCharacteristics {
    fn default() -> Self {
        Self {
            size: 0,
            dimensionality: 384, // Common embedding dimension
            intrinsic_dimensionality: 10.0,
            clustering_coefficient: 0.5,
            update_frequency: 0.0,
            memory_constraints: 8 * 1024 * 1024 * 1024, // 8GB default
        }
    }
}

// Index performance predictor based on theoretical models
pub struct IndexPredictor;

impl IndexPredictor {
    // Predict query latency for different index types
    pub fn predict_latency(
        index: &OptimalIndex,
        dataset_size: usize,
        dimensionality: usize,
        top_k: usize,
    ) -> f64 {
        match index {
            OptimalIndex::BruteForce => {
                // O(n * d) complexity
                let ops = dataset_size * dimensionality;
                ops as f64 * 1e-9 * 1000.0 // Convert to milliseconds (assuming 1ns per op)
            }
            OptimalIndex::HNSW { ef_search, .. } => {
                // O(log n * ef_search * d) complexity
                let log_n = (dataset_size as f64).log2();
                let ops = log_n * (*ef_search as f64) * (dimensionality as f64);
                ops * 2e-9 * 1000.0 // HNSW has some overhead
            }
            OptimalIndex::ProductQuantizedHNSW {
                ef_search, pq_m, ..
            } => {
                // Faster distance computation due to PQ
                let log_n = (dataset_size as f64).log2();
                let ops = log_n * (*ef_search as f64) * (*pq_m as f64); // Much smaller dimension
                ops * 1.5e-9 * 1000.0
            }
            OptimalIndex::HybridIndex { primary, fallback } => {
                // Average of both approaches
                let primary_lat =
                    Self::predict_latency(primary, dataset_size, dimensionality, top_k);
                let fallback_lat =
                    Self::predict_latency(fallback, dataset_size, dimensionality, top_k);
                (primary_lat + fallback_lat) * 0.5
            }
        }
    }

    // Predict memory usage
    pub fn predict_memory(
        index: &OptimalIndex,
        dataset_size: usize,
        dimensionality: usize,
    ) -> usize {
        let base_vectors = dataset_size * dimensionality * 4; // 4 bytes per f32

        match index {
            OptimalIndex::BruteForce => base_vectors,
            OptimalIndex::HNSW { m, .. } => {
                // HNSW graph overhead: approximately m connections per node
                let graph_overhead = dataset_size * m * 4; // 4 bytes per connection
                base_vectors + graph_overhead
            }
            OptimalIndex::ProductQuantizedHNSW { m, pq_m, .. } => {
                // PQ compressed vectors + graph
                let pq_vectors = dataset_size * pq_m; // 1 byte per subvector code
                let graph_overhead = dataset_size * m * 4;
                let codebook_size = pq_m * 256 * (dimensionality / pq_m) * 4; // Codebooks
                pq_vectors + graph_overhead + codebook_size
            }
            OptimalIndex::HybridIndex { primary, .. } => {
                Self::predict_memory(primary, dataset_size, dimensionality)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_stats() {
        let mut stats = WorkloadStats::new(100);

        // Simulate high-latency, high-frequency queries
        for _ in 0..50 {
            stats.record_query(15.0, 10, false); // 15ms latency
        }

        stats.dataset_characteristics.size = 100_000;
        stats.query_patterns.query_frequency = 2000.0;

        let recommended = stats.recommend_index();

        // Should recommend HNSW with optimized parameters due to high QPS and latency
        match recommended {
            OptimalIndex::HNSW { ef_search, .. } => {
                assert!(ef_search > 10); // Should use higher ef_search for better accuracy
            }
            _ => panic!("Expected HNSW recommendation"),
        }
    }

    #[test]
    fn test_index_predictor() {
        let hnsw_index = OptimalIndex::HNSW {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        };

        let latency = IndexPredictor::predict_latency(&hnsw_index, 100_000, 384, 10);
        let memory = IndexPredictor::predict_memory(&hnsw_index, 100_000, 384);

        assert!(latency > 0.0);
        assert!(memory > 0);
    }
}
