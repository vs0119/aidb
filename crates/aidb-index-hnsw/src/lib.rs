use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use aidb_core::{
    distance, Id, JsonValue, MetadataFilter, Metric, SearchResult, Vector, VectorIndex,
};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HnswBuildError {
    #[error("dimension must be greater than zero")]
    ZeroDimension,
    #[error("parameter m must be at least 2, got {0}")]
    InvalidM(usize),
    #[error("ef_construction must be >= m ({m}) and >= 2, got {ef_construction}")]
    InvalidEfConstruction { ef_construction: usize, m: usize },
    #[error("ef_search must be at least 1, got {0}")]
    InvalidEfSearch(usize),
}

#[derive(Default)]
struct VisitedScratch {
    marks: Vec<u32>,
    generation: u32,
}

impl VisitedScratch {
    fn start(&mut self, len: usize) {
        if self.marks.len() < len {
            self.marks.resize(len, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
    }

    fn mark(&mut self, idx: usize) -> bool {
        if idx >= self.marks.len() {
            self.marks.resize(idx + 1, 0);
        }
        if self.marks[idx] == self.generation {
            false
        } else {
            self.marks[idx] = self.generation;
            true
        }
    }
}

thread_local! {
    static VISITED_SCRATCH: RefCell<VisitedScratch> = RefCell::new(VisitedScratch::default());
    static CANDIDATE_BUF: RefCell<Vec<(usize, f32)>> = RefCell::new(Vec::new());
}

fn with_visited_scratch<R>(size: usize, f: impl FnOnce(&mut VisitedScratch) -> R) -> R {
    VISITED_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.start(size);
        f(&mut scratch)
    })
}

fn with_candidate_buf<R>(f: impl FnOnce(&mut Vec<(usize, f32)>) -> R) -> R {
    CANDIDATE_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        let res = f(&mut buf);
        buf.clear();
        res
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    id: Id,
    vector: Vector,
    payload: Option<JsonValue>,
    level: usize,
    neighbors: Vec<Vec<usize>>, // per level
    deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

impl HnswParams {
    pub fn validate(&self) -> Result<(), HnswBuildError> {
        if self.m < 2 {
            return Err(HnswBuildError::InvalidM(self.m));
        }
        if self.ef_construction < 2 || self.ef_construction < self.m {
            return Err(HnswBuildError::InvalidEfConstruction {
                ef_construction: self.ef_construction,
                m: self.m,
            });
        }
        if self.ef_search < 1 {
            return Err(HnswBuildError::InvalidEfSearch(self.ef_search));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndex {
    dim: usize,
    params: HnswParams,
    metric: Metric,
    nodes: Vec<Node>,
    id_to_idx: HashMap<Id, usize>,
    entry_point: Option<usize>,
    max_level: usize,
}

impl HnswIndex {
    pub fn try_new(dim: usize, metric: Metric, params: HnswParams) -> Result<Self, HnswBuildError> {
        if dim == 0 {
            return Err(HnswBuildError::ZeroDimension);
        }
        params.validate()?;

        Ok(Self {
            dim,
            params,
            metric,
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            entry_point: None,
            max_level: 0,
        })
    }

    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Self {
        Self::try_new(dim, metric, params).expect("invalid HNSW parameters")
    }

    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    fn random_level<R: Rng>(&self, rng: &mut R) -> usize {
        // Geometric distribution with p ~ 1/e
        let mut level = 0usize;
        while rng.gen::<f64>() < 1.0 / std::f64::consts::E {
            level += 1;
        }
        level
    }

    fn select_neighbors(&self, candidates: &mut Vec<(usize, f32)>, m: usize) -> Vec<usize> {
        if candidates.len() <= m {
            candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            return candidates.iter().map(|(i, _)| *i).collect();
        }

        let target = m.saturating_sub(1);
        candidates.select_nth_unstable_by(target, |a, b| a.1.total_cmp(&b.1));
        let best = &mut candidates[..=target];
        best.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        best.iter().map(|(i, _)| *i).collect()
    }

    fn link_bidirectional(&mut self, a: usize, b: usize, level: usize) {
        if self.nodes[a].neighbors[level].len() < self.params.m {
            self.nodes[a].neighbors[level].push(b);
        } else {
            // replace worst if b is closer
            let worst_idx = self.nodes[a].neighbors[level]
                .iter()
                .enumerate()
                .max_by(|(_, ia), (_, ib)| {
                    let da = distance(&self.nodes[a].vector, &self.nodes[**ia].vector, self.metric);
                    let db = distance(&self.nodes[a].vector, &self.nodes[**ib].vector, self.metric);
                    da.total_cmp(&db)
                })
                .map(|(i, _)| i);
            if let Some(wi) = worst_idx {
                let worst = self.nodes[a].neighbors[level][wi];
                let d_w = distance(
                    &self.nodes[a].vector,
                    &self.nodes[worst].vector,
                    self.metric,
                );
                let d_b = distance(&self.nodes[a].vector, &self.nodes[b].vector, self.metric);
                if d_b < d_w {
                    self.nodes[a].neighbors[level][wi] = b;
                }
            }
        }
        if self.nodes[b].neighbors[level].len() < self.params.m {
            self.nodes[b].neighbors[level].push(a);
        } else {
            let worst_idx = self.nodes[b].neighbors[level]
                .iter()
                .enumerate()
                .max_by(|(_, ia), (_, ib)| {
                    let da = distance(&self.nodes[b].vector, &self.nodes[**ia].vector, self.metric);
                    let db = distance(&self.nodes[b].vector, &self.nodes[**ib].vector, self.metric);
                    da.total_cmp(&db)
                })
                .map(|(i, _)| i);
            if let Some(wi) = worst_idx {
                let worst = self.nodes[b].neighbors[level][wi];
                let d_w = distance(
                    &self.nodes[b].vector,
                    &self.nodes[worst].vector,
                    self.metric,
                );
                let d_a = distance(&self.nodes[b].vector, &self.nodes[a].vector, self.metric);
                if d_a < d_w {
                    self.nodes[b].neighbors[level][wi] = a;
                }
            }
        }
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(usize, f32)> {
        #[derive(Copy, Clone)]
        struct Cand {
            idx: usize,
            dist: f32,
        }
        impl Eq for Cand {}
        impl PartialEq for Cand {
            fn eq(&self, other: &Self) -> bool {
                self.dist == other.dist
            }
        }
        impl Ord for Cand {
            fn cmp(&self, other: &Self) -> Ordering {
                other.dist.total_cmp(&self.dist)
            }
        }
        impl PartialOrd for Cand {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        with_visited_scratch(self.nodes.len(), |scratch| {
            let mut candidates = BinaryHeap::new();
            let mut top = BinaryHeap::new();

            let dist0 = distance(query, &self.nodes[entry].vector, self.metric);
            candidates.push(Cand {
                idx: entry,
                dist: dist0,
            });
            top.push(Cand {
                idx: entry,
                dist: dist0,
            });
            scratch.mark(entry);

            while let Some(c) = candidates.pop() {
                let worst_dist = top.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                if c.dist > worst_dist {
                    break;
                }
                for &n in &self.nodes[c.idx].neighbors[level] {
                    if !scratch.mark(n) {
                        continue;
                    }
                    if self.nodes[n].deleted {
                        continue;
                    }
                    let dn = distance(query, &self.nodes[n].vector, self.metric);
                    if top.len() < ef || dn < worst_dist {
                        candidates.push(Cand { idx: n, dist: dn });
                        top.push(Cand { idx: n, dist: dn });
                        if top.len() > ef {
                            top.pop();
                        }
                    }
                }
            }

            let res: Vec<(usize, f32)> = top
                .into_sorted_vec()
                .into_iter()
                .map(|c| (c.idx, c.dist))
                .collect();
            res
        })
    }
}

impl VectorIndex for HnswIndex {
    fn add(&mut self, id: Id, vector: Vector, payload: Option<JsonValue>) {
        debug_assert_eq!(vector.len(), self.dim);
        if let Some(&idx) = self.id_to_idx.get(&id) {
            // Prefer remove+reinsert to maintain graph quality
            self.nodes[idx].deleted = true;
        }

        let mut rng = rand::thread_rng();
        let level = self.random_level(&mut rng);
        let idx = self.nodes.len();
        let mut node = Node {
            id,
            vector,
            payload,
            level,
            neighbors: vec![Vec::new(); level + 1],
            deleted: false,
        };

        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_level = level;
            self.id_to_idx.insert(id, idx);
            self.nodes.push(node);
            return;
        }

        let mut ep = self.entry_point.unwrap();
        if level < self.max_level {
            // descend from max level to level+1 greedily
            for l in (level + 1..=self.max_level).rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    let mut best = ep;
                    let mut best_dist = distance(&self.nodes[ep].vector, &node.vector, self.metric);
                    for &n in &self.nodes[ep].neighbors[l] {
                        let d = distance(&self.nodes[n].vector, &node.vector, self.metric);
                        if d < best_dist {
                            best = n;
                            best_dist = d;
                            changed = true;
                        }
                    }
                    ep = best;
                }
            }
        }

        // From min(level, max_level) down to 0, perform ef_construction search and connect
        let start_level = std::cmp::min(level, self.max_level);
        for l in (0..=start_level).rev() {
            let neighs = with_candidate_buf(|buf| {
                buf.extend(
                    self.search_layer(&node.vector, ep, self.params.ef_construction, l)
                        .into_iter()
                        .filter(|(i, _)| !self.nodes[*i].deleted),
                );
                self.select_neighbors(buf, self.params.m)
            });

            // Insert node index placeholder to reference soon
            // (We push after linking to ensure idx exists in self.nodes)
            self.id_to_idx.insert(id, idx);

            // push so that references are valid
            if self.nodes.len() == idx {
                self.nodes.push(node);
            } else {
                self.nodes[idx] = node;
            }

            // link
            for &n in &neighs {
                self.link_bidirectional(idx, n, l);
            }

            // set ep for next lower layer as the closest among neighs or current ep
            if !neighs.is_empty() {
                ep = *neighs
                    .iter()
                    .min_by(|&&a, &&b| {
                        let da =
                            distance(&self.nodes[a].vector, &self.nodes[idx].vector, self.metric);
                        let db =
                            distance(&self.nodes[b].vector, &self.nodes[idx].vector, self.metric);
                        da.total_cmp(&db)
                    })
                    .unwrap();
            }

            // Prepare node for next loop iteration
            node = self.nodes[idx].clone();
        }

        if level > self.max_level {
            self.entry_point = Some(idx);
            self.max_level = level;
        }
    }

    fn remove(&mut self, id: &Id) -> bool {
        if let Some(&idx) = self.id_to_idx.get(id) {
            self.nodes[idx].deleted = true;
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
        if self.nodes.is_empty() {
            return Vec::new();
        }
        let mut ep = match self.entry_point {
            Some(e) => e,
            None => return Vec::new(),
        };
        // Greedy from top to level 1
        for l in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let mut best = ep;
                let mut best_dist = distance(&self.nodes[ep].vector, vector, metric);
                for &n in &self.nodes[ep].neighbors[l] {
                    let d = distance(&self.nodes[n].vector, vector, metric);
                    if d < best_dist {
                        best = n;
                        best_dist = d;
                        changed = true;
                    }
                }
                ep = best;
            }
        }

        let ef = self.params.ef_search.max(top_k);
        let res = self.search_layer(vector, ep, ef, 0);

        // Convert to SearchResult, apply filter, then take top_k
        let filt = filter.cloned();
        let mut out: Vec<(Id, f32, Option<JsonValue>)> = res
            .into_par_iter()
            .filter_map(|(i, d)| {
                let n = &self.nodes[i];
                if n.deleted {
                    return None;
                }
                if let Some(f) = &filt {
                    if !f.matches(&n.payload) {
                        return None;
                    }
                }
                Some((n.id, d, n.payload.clone()))
            })
            .collect();
        out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        out.truncate(top_k);
        out.into_iter()
            .map(|(id, score, payload)| SearchResult { id, score, payload })
            .collect()
    }

    fn len(&self) -> usize {
        self.nodes.iter().filter(|n| !n.deleted).count()
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

impl HnswIndex {
    pub fn rebuild(&mut self) {
        let mut fresh = HnswIndex::new(self.dim, self.metric, self.params.clone());
        // Preserve entry point by re-adding all non-deleted nodes
        let items: Vec<(Id, Vector, Option<JsonValue>)> = self
            .nodes
            .iter()
            .filter(|n| !n.deleted)
            .map(|n| (n.id, n.vector.clone(), n.payload.clone()))
            .collect();
        for (id, v, p) in items {
            fresh.add(id, v, p);
        }
        *self = fresh;
    }

    pub fn set_ef_search(&mut self, ef: usize) {
        self.params.ef_search = ef.max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidb_core::Metric;
    use serde_json::json;
    use uuid::Uuid;

    #[test]
    fn params_validation_catches_bad_configs() {
        let invalid_m = HnswParams {
            m: 1,
            ..Default::default()
        };
        assert!(matches!(
            HnswIndex::try_new(8, Metric::Cosine, invalid_m),
            Err(HnswBuildError::InvalidM(1))
        ));

        let invalid_efc = HnswParams {
            m: 16,
            ef_construction: 8,
            ef_search: 10,
        };
        assert!(matches!(
            HnswIndex::try_new(8, Metric::Cosine, invalid_efc),
            Err(HnswBuildError::InvalidEfConstruction { .. })
        ));

        let invalid_efs = HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 0,
        };
        assert!(matches!(
            HnswIndex::try_new(8, Metric::Cosine, invalid_efs),
            Err(HnswBuildError::InvalidEfSearch(0))
        ));

        assert!(matches!(
            HnswIndex::try_new(0, Metric::Cosine, HnswParams::default()),
            Err(HnswBuildError::ZeroDimension)
        ));
    }

    #[test]
    fn insert_update_delete_roundtrip() {
        let params = HnswParams {
            m: 8,
            ef_construction: 64,
            ef_search: 16,
        };
        let mut index = HnswIndex::try_new(4, Metric::Euclidean, params).expect("params valid");

        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();

        index.add(
            id_a,
            vec![0.9, 0.1, 0.2, 0.3],
            Some(json!({ "name": "a1" })),
        );
        index.add(
            id_b,
            vec![0.0, 1.0, 0.0, 0.0],
            Some(json!({ "name": "b1" })),
        );
        assert_eq!(index.len(), 2);

        let res = index.search(&[0.9, 0.1, 0.2, 0.3], 1, Metric::Euclidean, None);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, id_a);

        index.add(
            id_a,
            vec![0.8, 0.2, 0.1, 0.4],
            Some(json!({ "name": "a2" })),
        );
        assert_eq!(index.len(), 2);
        let res = index.search(&[0.8, 0.2, 0.1, 0.4], 1, Metric::Euclidean, None);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, id_a);
        assert_eq!(
            res[0]
                .payload
                .as_ref()
                .and_then(|p| p.get("name"))
                .and_then(|v| v.as_str()),
            Some("a2")
        );

        assert!(index.remove(&id_a));
        assert_eq!(index.len(), 1);

        let res_after_delete = index.search(&[0.8, 0.2, 0.1, 0.4], 1, Metric::Euclidean, None);
        assert!(res_after_delete.iter().all(|result| result.id != id_a));

        let res_b = index.search(&[0.0, 1.0, 0.0, 0.0], 1, Metric::Euclidean, None);
        assert_eq!(res_b.len(), 1);
        assert_eq!(res_b[0].id, id_b);
    }
}
