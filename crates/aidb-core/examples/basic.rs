use aidb_core::{Document, Id, Metric, SearchResult, VectorIndex};

struct MemoryIndex {
    dim: usize,
    docs: Vec<Document>,
}

impl MemoryIndex {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            docs: Vec::new(),
        }
    }
}

impl VectorIndex for MemoryIndex {
    fn add(&mut self, id: Id, vector: Vec<f32>, payload: Option<aidb_core::JsonValue>) {
        assert_eq!(vector.len(), self.dim);
        self.docs.push(Document {
            id,
            vector,
            payload,
        });
    }

    fn remove(&mut self, id: &Id) -> bool {
        let before = self.docs.len();
        self.docs.retain(|doc| &doc.id != id);
        before != self.docs.len()
    }

    fn search(
        &self,
        vector: &[f32],
        top_k: usize,
        metric: Metric,
        _filter: Option<&aidb_core::MetadataFilter>,
    ) -> Vec<SearchResult> {
        let mut results = self
            .docs
            .iter()
            .map(|doc| SearchResult {
                id: doc.id,
                score: 1.0 - aidb_core::distance(&doc.vector, vector, metric),
                payload: doc.payload.clone(),
            })
            .collect::<Vec<_>>();
        results.sort_by(|a, b| b.score.total_cmp(&a.score));
        results.truncate(top_k);
        results
    }

    fn len(&self) -> usize {
        self.docs.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

fn main() {
    let mut index = MemoryIndex::new(4);
    let doc_id = Id::new_v4();
    index.add(doc_id, vec![0.1, 0.2, 0.3, 0.4], None);

    let results = index.search(&[0.1, 0.2, 0.3, 0.4], 1, Metric::Cosine, None);
    println!("Found {} result(s)", results.len());
    assert!(results.iter().any(|r| r.id == doc_id));
}
