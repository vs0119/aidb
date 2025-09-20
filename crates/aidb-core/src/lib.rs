//! Core types and primitives for AIDB.
//!
//! The crate exposes vector types, distance metrics, and a trait that index
//! implementations must satisfy. Most consumers will interact with the
//! [`VectorIndex`] trait when embedding custom indexing strategies.
//!
//! # Examples
//!
//! ```
//! use aidb_core::{Document, Id, Metric, SearchResult, VectorIndex};
//!
//! struct MemoryIndex {
//!     dim: usize,
//!     docs: Vec<Document>,
//! }
//!
//! impl MemoryIndex {
//!     fn new(dim: usize) -> Self {
//!         Self { dim, docs: Vec::new() }
//!     }
//! }
//!
//! impl VectorIndex for MemoryIndex {
//!     fn add(&mut self, id: Id, vector: Vec<f32>, payload: Option<aidb_core::JsonValue>) {
//!         assert_eq!(vector.len(), self.dim);
//!         self.docs.push(Document { id, vector, payload });
//!     }
//!
//!     fn remove(&mut self, id: &Id) -> bool {
//!         let before = self.docs.len();
//!         self.docs.retain(|doc| &doc.id != id);
//!         before != self.docs.len()
//!     }
//!
//!     fn search(
//!         &self,
//!         vector: &[f32],
//!         top_k: usize,
//!         metric: Metric,
//!         _filter: Option<&aidb_core::MetadataFilter>,
//!     ) -> Vec<SearchResult> {
//!         let mut results = self
//!             .docs
//!             .iter()
//!             .map(|doc| SearchResult {
//!                 id: doc.id,
//!                 score: 1.0 - aidb_core::distance(&doc.vector, vector, metric),
//!                 payload: doc.payload.clone(),
//!             })
//!             .collect::<Vec<_>>();
//!         results.sort_by(|a, b| b.score.total_cmp(&a.score));
//!         results.truncate(top_k);
//!         results
//!     }
//!
//!     fn len(&self) -> usize {
//!         self.docs.len()
//!     }
//!
//!     fn dim(&self) -> usize {
//!         self.dim
//!     }
//! }
//!
//! let mut index = MemoryIndex::new(4);
//! let doc_id = Id::new_v4();
//! index.add(doc_id, vec![0.1, 0.2, 0.3, 0.4], None);
//! let results = index.search(&[0.1, 0.2, 0.3, 0.4], 1, Metric::Cosine, None);
//! assert_eq!(results.len(), 1);
//! assert_eq!(results[0].id, doc_id);
//! ```

pub type Vector = Vec<f32>;

pub use serde_json::Value as JsonValue;
pub use uuid::Uuid as Id;

pub mod adaptive;
pub mod quantization;
pub mod simd;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    Cosine,
    Euclidean,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Document {
    pub id: Id,
    pub vector: Vector,
    pub payload: Option<JsonValue>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct MetadataFilter {
    // Exact-match key/value filters
    pub equals: Vec<(String, JsonValue)>,
}

impl MetadataFilter {
    pub fn matches(&self, payload: &Option<JsonValue>) -> bool {
        if self.equals.is_empty() {
            return true;
        }
        let Some(p) = payload else { return false };
        for (k, v) in &self.equals {
            match p.get(k) {
                Some(val) if val == v => {}
                _ => return false,
            }
        }
        true
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub id: Id,
    pub score: f32,
    pub payload: Option<JsonValue>,
}

pub trait VectorIndex: Send + Sync {
    fn add(&mut self, id: Id, vector: Vector, payload: Option<JsonValue>);
    fn remove(&mut self, id: &Id) -> bool;
    fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        metric: Metric,
        filter: Option<&MetadataFilter>,
    ) -> Vec<SearchResult>;
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
}

#[inline]
pub fn distance(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => 1.0 - simd::simd_cosine_sim(a, b),
        Metric::Euclidean => simd::simd_l2_distance(a, b),
    }
}

// Fallback non-SIMD functions for compatibility
#[inline]
pub fn distance_scalar(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => 1.0 - cosine_sim(a, b),
        Metric::Euclidean => l2(a, b),
    }
}

#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        let x = a[i];
        let y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_filter_matches_empty() {
        let filter = MetadataFilter::default();
        assert!(filter.matches(&None));
        assert!(filter.matches(&Some(JsonValue::Null)));
    }

    #[test]
    fn metadata_filter_exact_match() {
        let mut filter = MetadataFilter::default();
        filter
            .equals
            .push(("kind".to_string(), JsonValue::String("doc".into())));
        let payload = serde_json::json!({"kind": "doc", "title": "rust"});
        assert!(filter.matches(&Some(payload)));
    }

    #[test]
    fn metadata_filter_rejects_missing_keys() {
        let mut filter = MetadataFilter::default();
        filter
            .equals
            .push(("kind".to_string(), JsonValue::String("doc".into())));
        let payload = serde_json::json!({"other": "value"});
        assert!(!filter.matches(&Some(payload)));
    }

    #[test]
    fn cosine_similarity_respects_zero_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_sim(&a, &b), 0.0);
    }

    #[test]
    fn l2_distance_is_symmetric() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((l2(&a, &b) - l2(&b, &a)).abs() < f32::EPSILON);
    }
}
