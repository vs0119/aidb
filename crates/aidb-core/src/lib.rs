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
