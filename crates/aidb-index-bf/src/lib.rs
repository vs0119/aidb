use aidb_core::{
    distance, Id, JsonValue, MetadataFilter, Metric, SearchResult, Vector, VectorIndex,
};
use rayon::prelude::*;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BruteForceIndex {
    dim: usize,
    // id, vector, payload
    entries: Vec<(Id, Vector, Option<JsonValue>)>,
}

impl BruteForceIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
        }
    }
}

impl VectorIndex for BruteForceIndex {
    fn add(&mut self, id: Id, vector: Vector, payload: Option<JsonValue>) {
        debug_assert_eq!(vector.len(), self.dim);
        // replace if exists
        if let Some(i) = self.entries.iter().position(|(eid, _, _)| eid == &id) {
            self.entries[i] = (id, vector, payload);
        } else {
            self.entries.push((id, vector, payload));
        }
    }

    fn remove(&mut self, id: &Id) -> bool {
        if let Some(i) = self.entries.iter().position(|(eid, _, _)| eid == id) {
            self.entries.swap_remove(i);
            true
        } else {
            false
        }
    }

    fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        metric: Metric,
        filter: Option<&MetadataFilter>,
    ) -> Vec<SearchResult> {
        let filt = filter.cloned();
        let mut scored: Vec<(Id, f32, Option<JsonValue>)> = self
            .entries
            .par_iter()
            .filter_map(|(id, v, p)| {
                if let Some(f) = &filt {
                    if !f.matches(p) {
                        return None;
                    }
                }
                let d = distance(vector, v, metric);
                Some((*id, d, p.clone()))
            })
            .collect();

        // For cosine similarity we converted to distance = 1 - sim.
        // Lower is better for both metrics. Sort ascending.
        let k = top_k.min(scored.len());
        // partial sort for top_k
        if k < scored.len() {
            scored.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
            scored[..k].sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            scored.truncate(k);
        } else {
            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        }

        scored
            .into_iter()
            .map(|(id, dist, payload)| SearchResult {
                id,
                score: dist,
                payload,
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
    fn dim(&self) -> usize {
        self.dim
    }
}
