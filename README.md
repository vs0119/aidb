# AIDB — AI-Optimized Vector Database (Rust)

AIDB is an experimental, high-performance database engine tailored for AI workloads:
- Vector similarity search (cosine, L2) with metadata filtering
- Durable writes via WAL; fast in-memory read path
- Modular index trait with a pluggable ANN layer (starts with brute-force; HNSW/IVF planned)
- Minimal HTTP server for CRUD + search

This repository is a scaffold that runs locally and establishes clean boundaries for storage, indexing, and serving. It’s designed to evolve into an extreme-performance engine by swapping the index and storage backends without rewriting the API surface.

## Quick Overview

- `crates/aidb-core`: Core types, metrics, index trait
- `crates/aidb-index-bf`: Brute-force index with parallel scoring
- `crates/aidb-storage`: WAL, in-memory collection, simple engine
- `crates/aidb-server`: Axum HTTP server exposing collection + search APIs

## Building & Running

Rust 1.70+ recommended. The first build fetches dependencies from crates.io.

```
# from repo root
cargo run -p aidb-server
# server listens on 0.0.0.0:8080
```

### Benchmarks

- In‑process index benchmark (no HTTP):
```
cargo run -p aidb-bench --release -- --mode=index --index=hnsw --dim=768 --n=200000 --q=500 --topk=10 --m=16 --efc=200 --efs=64
```

- HTTP end‑to‑end benchmark (server must be running):
```
cargo run -p aidb-bench --release -- --mode=http --base=http://127.0.0.1:8080 --col=bench --index=hnsw --dim=768 --n=200000 --q=500 --topk=10 --m=16 --efc=200 --efs=64 --batch=1000
```

- Reporting:
```
# JSON to stdout
cargo run -p aidb-bench --release -- --mode=index --report=json

# CSV to file
cargo run -p aidb-bench --release -- --mode=http --report=csv --out=bench.csv
```

## API

- `POST /collections` – create a collection
```json
{ "name": "docs", "dim": 768, "metric": "cosine", "wal_dir": "./data" }
```

- `POST /collections/:name/points` – upsert a vector
```json
{ "vector": [0.12, 0.98, ...], "payload": {"source": "a.txt"} }
```

- `POST /collections/:name/search` – vector search
```json
{ "vector": [0.1, 0.2, ...], "top_k": 10, "filter": {"source": "a.txt"} }
```
- `POST /collections/:name/search:batch` – batch vector search (single `top_k`/filter shared)
```json
{
  "queries": [[0.1, 0.2, ...], [0.9, 0.1, ...]],
  "top_k": 10,
  "filter": {"source": "a.txt"}
}
```

- `DELETE /collections/:name/points/:id` – delete a point

## Performance Roadmap

This scaffold favors correctness and composability. To reach extreme performance:

- Indexing
  - HNSW with visited-set pruning, heuristics, multi-layer graph
  - IVF-Flat, IVF-PQ with training/residuals; OPQ for rotation
  - Scalar quantization + Product quantization; mixed-precision scoring
  - Batched, SIMD-accelerated distance kernels; memory alignment
  - Hybrid search (BM25 + vector) via inverted lists + re-rank

- Storage
  - Memory-mapped segments; zero-copy vector slices
  - LSM-tree with compaction filters and tombstone GC
  - Background index builds and atomic index swaps
  - Snapshots + incremental checkpoints; CRC + fencing tokens

- Serving
  - Sharding + replication (Raft/Quorum); consistent hashing
  - Columnar metadata store with predicate pushdown
  - Streaming ingestion; backpressure and rate limiting
  - Observability: request tracing, metrics, p99 SLAs

- Hardware Acceleration
  - GPU ANN (cuVS/RAFT), CPU AVX512/NEON kernels
  - NUMA-aware scheduling; core pinning; cache locality

## Notes

- The brute-force index is parallelized (Rayon) and good for small/mid datasets and testing; replace with ANN for large scale.
- WAL provides durability; recovery replays ops into the index.
- Filters support exact-match key/value pairs for now.

## License

Proprietary scaffolding for local development unless you choose otherwise.
