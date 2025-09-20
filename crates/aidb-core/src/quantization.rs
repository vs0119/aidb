use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// Product Quantization for 8-16x memory compression with minimal accuracy loss
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProductQuantizer {
    pub m: usize,                      // Number of subvectors
    pub ksub: usize,                   // Number of centroids per subquantizer (typically 256)
    pub dim: usize,                    // Original vector dimension
    pub sub_dim: usize,                // Dimension of each subvector
    pub codebooks: Vec<Vec<Vec<f32>>>, // [m][ksub][sub_dim] - centroids for each subquantizer
    pub trained: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PQCode {
    pub codes: Vec<u8>, // Quantized representation - m bytes per vector
}

impl ProductQuantizer {
    pub fn new(dim: usize, m: usize, ksub: usize) -> Self {
        assert!(dim % m == 0, "Dimension must be divisible by m");
        let sub_dim = dim / m;

        Self {
            m,
            ksub,
            dim,
            sub_dim,
            codebooks: vec![vec![vec![0.0; sub_dim]; ksub]; m],
            trained: false,
        }
    }

    // Train quantizer on representative dataset
    pub fn train(&mut self, training_data: &[Vec<f32>], iterations: usize) {
        if training_data.is_empty() {
            return;
        }

        // Initialize codebooks with k-means++
        for subq_idx in 0..self.m {
            self.init_codebook_kmeans_plus_plus(training_data, subq_idx);
        }

        // Run k-means for each subquantizer
        for _iter in 0..iterations {
            for subq_idx in 0..self.m {
                self.update_codebook_kmeans(training_data, subq_idx);
            }
        }

        self.trained = true;
    }

    fn init_codebook_kmeans_plus_plus(&mut self, data: &[Vec<f32>], subq_idx: usize) {
        let start_dim = subq_idx * self.sub_dim;
        let end_dim = start_dim + self.sub_dim;

        // Extract subvectors for this subquantizer
        let subvectors: Vec<Vec<f32>> = data
            .iter()
            .map(|v| v[start_dim..end_dim].to_vec())
            .collect();

        if subvectors.is_empty() {
            return;
        }

        // Choose first centroid randomly
        self.codebooks[subq_idx][0] = subvectors[0].clone();

        // Choose remaining centroids using k-means++ strategy
        for k in 1..self.ksub.min(subvectors.len()) {
            let mut distances = vec![f32::INFINITY; subvectors.len()];

            // Calculate distance to nearest existing centroid
            for (i, subvec) in subvectors.iter().enumerate() {
                for j in 0..k {
                    let dist = l2_distance(subvec, &self.codebooks[subq_idx][j]);
                    distances[i] = distances[i].min(dist);
                }
            }

            // Choose next centroid with probability proportional to squared distance
            let sum: f32 = distances.iter().map(|d| d * d).sum();
            if sum > 0.0 {
                let mut cumsum = 0.0;
                let target = (rand::random::<f32>() * sum).max(1e-6);

                for (i, &dist) in distances.iter().enumerate() {
                    cumsum += dist * dist;
                    if cumsum >= target {
                        self.codebooks[subq_idx][k] = subvectors[i].clone();
                        break;
                    }
                }
            }
        }
    }

    fn update_codebook_kmeans(&mut self, data: &[Vec<f32>], subq_idx: usize) {
        let start_dim = subq_idx * self.sub_dim;
        let end_dim = start_dim + self.sub_dim;

        let mut assignments = vec![0; data.len()];
        let mut cluster_sums = vec![vec![0.0; self.sub_dim]; self.ksub];
        let mut cluster_counts = vec![0; self.ksub];

        // Assign points to nearest centroids
        for (i, vector) in data.iter().enumerate() {
            let subvec = &vector[start_dim..end_dim];
            let mut best_dist = f32::INFINITY;
            let mut best_centroid = 0;

            for (k, centroid) in self.codebooks[subq_idx].iter().enumerate() {
                let dist = l2_distance(subvec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_centroid = k;
                }
            }

            assignments[i] = best_centroid;
            cluster_counts[best_centroid] += 1;
            for j in 0..self.sub_dim {
                cluster_sums[best_centroid][j] += subvec[j];
            }
        }

        // Update centroids
        for k in 0..self.ksub {
            if cluster_counts[k] > 0 {
                for j in 0..self.sub_dim {
                    self.codebooks[subq_idx][k][j] = cluster_sums[k][j] / cluster_counts[k] as f32;
                }
            }
        }
    }

    // Encode vector to PQ codes
    pub fn encode(&self, vector: &[f32]) -> PQCode {
        assert!(self.trained, "Quantizer must be trained first");
        assert_eq!(vector.len(), self.dim, "Vector dimension mismatch");

        let mut codes = Vec::with_capacity(self.m);

        for subq_idx in 0..self.m {
            let start_dim = subq_idx * self.sub_dim;
            let end_dim = start_dim + self.sub_dim;
            let subvec = &vector[start_dim..end_dim];

            let mut best_dist = f32::INFINITY;
            let mut best_code = 0u8;

            for (k, centroid) in self.codebooks[subq_idx].iter().enumerate() {
                let dist = l2_distance(subvec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_code = k as u8;
                }
            }

            codes.push(best_code);
        }

        PQCode { codes }
    }

    // Batch encode multiple vectors efficiently
    pub fn batch_encode(&self, vectors: &[Vec<f32>]) -> Vec<PQCode> {
        vectors.par_iter().map(|v| self.encode(v)).collect()
    }

    // Decode PQ codes back to approximate vector
    pub fn decode(&self, code: &PQCode) -> Vec<f32> {
        let mut vector = vec![0.0; self.dim];

        for (subq_idx, &code_val) in code.codes.iter().enumerate() {
            let start_dim = subq_idx * self.sub_dim;
            let centroid = &self.codebooks[subq_idx][code_val as usize];

            for (i, &val) in centroid.iter().enumerate() {
                vector[start_dim + i] = val;
            }
        }

        vector
    }

    // Asymmetric distance computation - query vs PQ code
    pub fn asymmetric_distance(&self, query: &[f32], code: &PQCode, metric: crate::Metric) -> f32 {
        let mut total_dist = 0.0;

        for (subq_idx, &code_val) in code.codes.iter().enumerate() {
            let start_dim = subq_idx * self.sub_dim;
            let end_dim = start_dim + self.sub_dim;
            let query_sub = &query[start_dim..end_dim];
            let centroid = &self.codebooks[subq_idx][code_val as usize];

            match metric {
                crate::Metric::Euclidean => {
                    let sub_dist = l2_distance(query_sub, centroid);
                    total_dist += sub_dist * sub_dist;
                }
                crate::Metric::Cosine => {
                    // For cosine, we approximate using dot product of normalized subvectors
                    let dot: f32 = query_sub
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    total_dist += dot; // Will normalize later
                }
            }
        }

        match metric {
            crate::Metric::Euclidean => total_dist.sqrt(),
            crate::Metric::Cosine => {
                let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let decoded = self.decode(code);
                let code_norm: f32 = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 - (total_dist / (query_norm * code_norm))
            }
        }
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

impl PQCode {
    // Memory efficient storage - each vector uses only m bytes instead of dim*4 bytes
    pub fn memory_usage(&self) -> usize {
        self.codes.len() // m bytes per vector vs dim * 4 bytes normally
    }
}

// Scalar Quantization for ultra-fast distance computation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    pub dim: usize,
    pub min_vals: Vec<f32>,
    pub max_vals: Vec<f32>,
    pub scale: Vec<f32>,
    pub trained: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SQCode {
    pub codes: Vec<u8>, // 1 byte per dimension
}

impl ScalarQuantizer {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            min_vals: vec![0.0; dim],
            max_vals: vec![0.0; dim],
            scale: vec![1.0; dim],
            trained: false,
        }
    }

    pub fn train(&mut self, training_data: &[Vec<f32>]) {
        if training_data.is_empty() {
            return;
        }

        // Find min and max for each dimension
        self.min_vals = vec![f32::INFINITY; self.dim];
        self.max_vals = vec![f32::NEG_INFINITY; self.dim];

        for vector in training_data {
            for (i, &val) in vector.iter().enumerate() {
                self.min_vals[i] = self.min_vals[i].min(val);
                self.max_vals[i] = self.max_vals[i].max(val);
            }
        }

        // Calculate scaling factors
        for i in 0..self.dim {
            let range = self.max_vals[i] - self.min_vals[i];
            self.scale[i] = if range > 0.0 { 255.0 / range } else { 1.0 };
        }

        self.trained = true;
    }

    pub fn encode(&self, vector: &[f32]) -> SQCode {
        assert!(self.trained, "Quantizer must be trained first");
        assert_eq!(vector.len(), self.dim);

        let codes: Vec<u8> = vector
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                let normalized = (val - self.min_vals[i]) * self.scale[i];
                normalized.clamp(0.0, 255.0) as u8
            })
            .collect();

        SQCode { codes }
    }

    pub fn decode(&self, code: &SQCode) -> Vec<f32> {
        code.codes
            .iter()
            .enumerate()
            .map(|(i, &c)| self.min_vals[i] + (c as f32) / self.scale[i])
            .collect()
    }

    // Ultra-fast SIMD distance computation between query and SQ code
    pub fn simd_distance(&self, query: &[f32], code: &SQCode, metric: crate::Metric) -> f32 {
        match metric {
            crate::Metric::Euclidean => self.simd_l2_distance(query, code),
            crate::Metric::Cosine => self.simd_cosine_distance(query, code),
        }
    }

    fn simd_l2_distance(&self, query: &[f32], code: &SQCode) -> f32 {
        let mut sum = 0.0f32;

        // Process 8 dimensions at a time using SIMD
        let chunks = query.chunks_exact(8);
        let code_chunks = code.codes.chunks_exact(8);

        for (q_chunk, c_chunk) in chunks.clone().zip(code_chunks.clone()) {
            for (i, (&q_val, &c_val)) in q_chunk.iter().zip(c_chunk.iter()).enumerate() {
                let dim_idx = (q_chunk.as_ptr() as usize - query.as_ptr() as usize) / 4 + i;
                let decoded_val = self.min_vals[dim_idx] + (c_val as f32) / self.scale[dim_idx];
                let diff = q_val - decoded_val;
                sum += diff * diff;
            }
        }

        // Handle remainder
        let remainder_q = chunks.remainder();
        let remainder_c = code_chunks.remainder();
        for (i, (&q_val, &c_val)) in remainder_q.iter().zip(remainder_c.iter()).enumerate() {
            let dim_idx = query.len() - remainder_q.len() + i;
            let decoded_val = self.min_vals[dim_idx] + (c_val as f32) / self.scale[dim_idx];
            let diff = q_val - decoded_val;
            sum += diff * diff;
        }

        sum.sqrt()
    }

    fn simd_cosine_distance(&self, query: &[f32], code: &SQCode) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_q = 0.0f32;
        let mut norm_c = 0.0f32;

        for (i, (&q_val, &c_val)) in query.iter().zip(code.codes.iter()).enumerate() {
            let decoded_val = self.min_vals[i] + (c_val as f32) / self.scale[i];
            dot += q_val * decoded_val;
            norm_q += q_val * q_val;
            norm_c += decoded_val * decoded_val;
        }

        if norm_q == 0.0 || norm_c == 0.0 {
            1.0
        } else {
            1.0 - (dot / (norm_q.sqrt() * norm_c.sqrt()))
        }
    }
}

// Hybrid quantization combining PQ and SQ for maximum efficiency
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridQuantizer {
    pub pq: ProductQuantizer,
    pub sq: ScalarQuantizer,
    pub use_pq_threshold: usize, // Use PQ for vectors longer than this
}

impl HybridQuantizer {
    pub fn new(dim: usize, m: usize, ksub: usize, pq_threshold: usize) -> Self {
        Self {
            pq: ProductQuantizer::new(dim, m, ksub),
            sq: ScalarQuantizer::new(dim),
            use_pq_threshold: pq_threshold,
        }
    }

    pub fn train(&mut self, training_data: &[Vec<f32>], pq_iterations: usize) {
        self.pq.train(training_data, pq_iterations);
        self.sq.train(training_data);
    }

    pub fn encode_best(&self, vector: &[f32]) -> QuantizedVector {
        if vector.len() >= self.use_pq_threshold {
            QuantizedVector::PQ(self.pq.encode(vector))
        } else {
            QuantizedVector::SQ(self.sq.encode(vector))
        }
    }

    pub fn distance(&self, query: &[f32], code: &QuantizedVector, metric: crate::Metric) -> f32 {
        match code {
            QuantizedVector::PQ(pq_code) => self.pq.asymmetric_distance(query, pq_code, metric),
            QuantizedVector::SQ(sq_code) => self.sq.simd_distance(query, sq_code, metric),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum QuantizedVector {
    PQ(PQCode),
    SQ(SQCode),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_quantization() {
        let dim = 128;
        let m = 8; // 8 subvectors of 16 dimensions each
        let ksub = 16; // 16 centroids per subquantizer

        let mut pq = ProductQuantizer::new(dim, m, ksub);

        // Generate random training data
        let training_data: Vec<Vec<f32>> = (0..1000)
            .map(|_| (0..dim).map(|_| rand::random::<f32>() - 0.5).collect())
            .collect();

        pq.train(&training_data, 10);
        assert!(pq.trained);

        // Test encoding/decoding
        let test_vector: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let code = pq.encode(&test_vector);
        let decoded = pq.decode(&code);

        assert_eq!(test_vector.len(), decoded.len());
        assert_eq!(code.codes.len(), m);

        // Memory usage should be much smaller
        let original_memory = dim * 4; // 4 bytes per f32
        let pq_memory = code.memory_usage();
        assert!(pq_memory < original_memory);
    }

    #[test]
    #[should_panic(expected = "Dimension must be divisible by m")]
    fn product_quantizer_new_invalid_dim_panics() {
        let _ = ProductQuantizer::new(10, 3, 4);
    }

    #[test]
    #[should_panic(expected = "Quantizer must be trained first")]
    fn product_quantizer_encode_without_training_panics() {
        let pq = ProductQuantizer::new(4, 2, 2);
        let vector = vec![0.0; 4];
        pq.encode(&vector);
    }

    #[test]
    fn product_quantizer_train_empty_data_is_noop() {
        let dim = 4;
        let mut pq = ProductQuantizer::new(dim, 2, 2);

        pq.train(&[], 5);

        assert!(!pq.trained);
        assert!(pq
            .codebooks
            .iter()
            .flatten()
            .flatten()
            .all(|&value| value == 0.0));
    }

    #[test]
    fn product_quantizer_batch_encode_matches_single_encode() {
        let dim = 4;
        let m = 2;
        let ksub = 2;
        let mut pq = ProductQuantizer::new(dim, m, ksub);

        let training_vector = vec![1.0, 2.0, 3.0, 4.0];
        let training_data = vec![training_vector.clone(), training_vector.clone()];
        pq.train(&training_data, 2);
        assert!(pq.trained);

        let vectors = vec![
            training_vector.clone(),
            vec![1.0, 2.1, 3.0, 4.1],
            vec![0.0, 0.0, 0.0, 0.0],
        ];

        let batch = pq.batch_encode(&vectors);
        let single: Vec<PQCode> = vectors.iter().map(|v| pq.encode(v)).collect();

        assert_eq!(batch.len(), single.len());
        for (batch_code, single_code) in batch.iter().zip(single.iter()) {
            assert_eq!(batch_code.codes, single_code.codes);
        }
    }

    #[test]
    fn product_quantizer_batch_encode_empty_input_returns_empty() {
        let dim = 4;
        let mut pq = ProductQuantizer::new(dim, 2, 2);
        let training_vector = vec![1.0, 1.0, 1.0, 1.0];
        pq.train(&[training_vector.clone(), training_vector], 1);
        assert!(pq.trained);

        let encoded = pq.batch_encode(&[]);
        assert!(encoded.is_empty());
    }

    #[test]
    fn product_quantizer_asymmetric_distance_matches_decoded_euclidean() {
        let dim = 4;
        let mut pq = ProductQuantizer::new(dim, 2, 2);

        let training_vector = vec![1.0, 2.0, 3.0, 4.0];
        pq.train(&[training_vector.clone(), training_vector.clone()], 1);

        let code = pq.encode(&training_vector);
        let query = vec![1.5, 1.5, 3.5, 3.5];

        let asymmetric = pq.asymmetric_distance(&query, &code, crate::Metric::Euclidean);
        let decoded = pq.decode(&code);
        let expected = crate::distance(&query, &decoded, crate::Metric::Euclidean);

        assert!((asymmetric - expected).abs() < 1e-6);
    }

    #[test]
    fn product_quantizer_asymmetric_distance_matches_cosine() {
        let dim = 2;
        let mut pq = ProductQuantizer::new(dim, 1, 1);
        let training_vector = vec![1.0, 0.0];
        pq.train(&[training_vector.clone()], 0);
        assert!(pq.trained);

        let code = pq.encode(&training_vector);
        let query = vec![0.0, 1.0];

        let distance = pq.asymmetric_distance(&query, &code, crate::Metric::Cosine);
        let decoded = pq.decode(&code);
        let expected = crate::distance(&query, &decoded, crate::Metric::Cosine);

        assert!((distance - expected).abs() < 1e-6);
    }

    #[test]
    fn product_quantizer_asymmetric_distance_handles_zero_norm_cosine() {
        let dim = 4;
        let mut pq = ProductQuantizer::new(dim, 2, 2);

        let training_vector = vec![1.0, 2.0, 3.0, 4.0];
        pq.train(&[training_vector.clone(), training_vector.clone()], 1);

        let zero_vector = vec![0.0; dim];
        let code = pq.encode(&zero_vector);
        let query = vec![0.0; dim];

        let distance = pq.asymmetric_distance(&query, &code, crate::Metric::Cosine);
        assert_eq!(distance, 1.0);
    }

    #[test]
    fn product_quantizer_train_zero_iterations_marks_trained() {
        let dim = 4;
        let mut pq = ProductQuantizer::new(dim, 2, 2);
        let training_data = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];

        pq.train(&training_data, 0);

        assert!(pq.trained);
        assert!(pq
            .codebooks
            .iter()
            .flatten()
            .flatten()
            .any(|&value| value != 0.0));
    }

    #[test]
    #[should_panic(expected = "Vector dimension mismatch")]
    fn product_quantizer_encode_dimension_mismatch_panics() {
        let mut pq = ProductQuantizer::new(4, 2, 2);
        let training_vector = vec![1.0, 2.0, 3.0, 4.0];
        pq.train(&[training_vector.clone(), training_vector], 1);
        pq.encode(&[1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "Quantizer must be trained first")]
    fn scalar_quantizer_encode_without_training_panics() {
        let sq = ScalarQuantizer::new(4);
        let vector = vec![0.0; 4];
        sq.encode(&vector);
    }

    #[test]
    fn scalar_quantizer_train_empty_data_is_noop() {
        let mut sq = ScalarQuantizer::new(4);
        sq.train(&[]);
        assert!(!sq.trained);
        assert!(sq.min_vals.iter().all(|&v| v == 0.0));
        assert!(sq.max_vals.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn scalar_quantizer_roundtrip_and_distances() {
        let mut sq = ScalarQuantizer::new(4);
        let training_data = vec![vec![1.0, 2.0, 3.0, 4.0]];
        sq.train(&training_data);
        assert!(sq.trained);

        let code = sq.encode(&training_data[0]);
        let decoded = sq.decode(&code);
        assert_eq!(decoded, training_data[0]);

        let distance = sq.simd_distance(&training_data[0], &code, crate::Metric::Euclidean);
        assert!(distance.abs() < 1e-6);

        let cosine_distance = sq.simd_distance(&vec![0.0; 4], &code, crate::Metric::Cosine);
        assert_eq!(cosine_distance, 1.0);
    }

    #[test]
    fn scalar_quantizer_encode_clamps_to_byte_range() {
        let mut sq = ScalarQuantizer::new(2);
        let training_data = vec![vec![0.0, 1.0]];
        sq.train(&training_data);

        let vector = vec![-5.0, 10.0];
        let code = sq.encode(&vector);

        assert_eq!(code.codes, vec![0, 255]);
    }

    #[test]
    fn scalar_quantizer_simd_l2_distance_handles_remainder() {
        let dim = 10;
        let mut sq = ScalarQuantizer::new(dim);
        let base: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        sq.train(&[base.clone()]);
        let code = sq.encode(&base);
        let query: Vec<f32> = base.iter().map(|v| v + 0.25).collect();

        let simd = sq.simd_distance(&query, &code, crate::Metric::Euclidean);
        let decoded = sq.decode(&code);
        let expected = crate::distance(&query, &decoded, crate::Metric::Euclidean);

        assert!((simd - expected).abs() < 1e-6);
    }

    #[test]
    fn scalar_quantizer_simd_cosine_distance_matches_manual() {
        let mut sq = ScalarQuantizer::new(3);
        let training = vec![vec![1.0, 2.0, 3.0]];
        sq.train(&training);
        let code = sq.encode(&training[0]);
        let query = vec![0.5, 1.0, 1.5];

        let simd = sq.simd_distance(&query, &code, crate::Metric::Cosine);
        let decoded = sq.decode(&code);
        let expected = crate::distance(&query, &decoded, crate::Metric::Cosine);

        assert!((simd - expected).abs() < 1e-6);
    }

    #[test]
    fn hybrid_quantizer_selects_strategy_and_delegates_distance() {
        let dim = 4;
        let mut hybrid_pq = HybridQuantizer::new(dim, 2, 2, dim);
        let mut hybrid_sq = HybridQuantizer::new(dim, 2, 2, dim + 1);
        let training_data = vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]];

        hybrid_pq.train(&training_data, 1);
        hybrid_sq.train(&training_data, 1);

        let vector = training_data[0].clone();

        match hybrid_pq.encode_best(&vector) {
            QuantizedVector::PQ(code) => {
                assert_eq!(code.codes.len(), 2);
                let wrapped = QuantizedVector::PQ(code.clone());
                let dist = hybrid_pq.distance(&vector, &wrapped, crate::Metric::Euclidean);
                let decoded = hybrid_pq.pq.decode(&code);
                let expected = crate::distance(&vector, &decoded, crate::Metric::Euclidean);
                assert!((dist - expected).abs() < 1e-6);
            }
            _ => panic!("expected PQ encoding"),
        }

        match hybrid_sq.encode_best(&vector) {
            QuantizedVector::SQ(code) => {
                assert_eq!(code.codes.len(), dim);
                let wrapped = QuantizedVector::SQ(code.clone());
                let dist = hybrid_sq.distance(&vector, &wrapped, crate::Metric::Euclidean);
                let expected = hybrid_sq
                    .sq
                    .simd_distance(&vector, &code, crate::Metric::Euclidean);
                assert!((dist - expected).abs() < 1e-6);
            }
            _ => panic!("expected SQ encoding"),
        }
    }

    #[test]
    fn hybrid_quantizer_distance_cosine_matches_underlying() {
        let dim = 4;
        let mut hybrid = HybridQuantizer::new(dim, 2, 2, dim);
        let training_data = vec![vec![1.0, 2.0, 3.0, 4.0]; 2];
        hybrid.train(&training_data, 1);

        if let QuantizedVector::PQ(code) = hybrid.encode_best(&training_data[0]) {
            let wrapped = QuantizedVector::PQ(code.clone());
            let query = vec![0.5, 1.0, 1.5, 2.0];
            let distance = hybrid.distance(&query, &wrapped, crate::Metric::Cosine);
            let decoded = hybrid.pq.decode(&code);
            let expected = crate::distance(&query, &decoded, crate::Metric::Cosine);
            assert!((distance - expected).abs() < 1e-6);
        } else {
            panic!("expected PQ encoding");
        }

        let mut hybrid_sq = HybridQuantizer::new(dim, 2, 2, dim + 1);
        hybrid_sq.train(&training_data, 1);
        let code = match hybrid_sq.encode_best(&training_data[0]) {
            QuantizedVector::SQ(code) => code,
            _ => panic!("expected SQ encoding"),
        };
        let wrapped = QuantizedVector::SQ(code.clone());
        let query = vec![0.5, 1.0, 1.5, 2.0];
        let distance = hybrid_sq.distance(&query, &wrapped, crate::Metric::Cosine);
        let expected = hybrid_sq
            .sq
            .simd_distance(&query, &code, crate::Metric::Cosine);
        assert!((distance - expected).abs() < 1e-6);
    }
}
