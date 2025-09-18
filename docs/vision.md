# AIDB Vision and Architecture Notes

This document captures the current foundation and the remaining roadmap toward a best‑in‑class AI database.

## What Exists (from your storage engine)
- 8KB PostgreSQL‑style pages with slot directories, free space mgmt, checksums, LSN tracking
- MVCC with 4 isolation levels, snapshot isolation, deadlock detection
- ARC‑inspired buffer pool with pin/unpin and hit ratio tracking
- Multi‑level compression (LZ4, Zstd, Delta, Dictionary) with auto selection
- Auto‑vacuum for dead row cleanup, cost‑based throttling

## What Exists (from this repo)
- Vector DB scaffold in Rust: core, storage (WAL), indexes (BF, HNSW MVP), HTTP server
- Benchmarks: in‑process and HTTP, JSON/CSV output
- Admin features: batch ingest, stats, snapshot, compact, HNSW params update

## Remaining High‑Impact Work
- Indexing: HNSW robustness, IVF‑Flat/PQ(+OPQ), SIMD kernels, batched search
- Transactional index integration with MVCC (xmin/xmax visibility, rollback safety)
- Storage: segment files, mmap read path, snapshots + incremental checkpoints, WAL truncation
- Query planner with hybrid retrieval (BM25 + vector re‑rank), predicate pushdown
- Serving: gRPC, per‑query knobs (ef_search/nprobe), auth, metrics, tracing
- Distribution: replication, sharding, rebalancing
- Filtering: typed columns, range/set predicates, bitmap/columnar metadata store
- Reliability: crash consistency tests, fuzzing, soak tests

## Immediate Next Steps
1) Native macOS UI for AIDB admin and querying
2) Transactional hooks between MVCC pages and vector indexes
3) SIMD distance kernels and ANN benchmarks parity

This file serves as a living record to align development.
