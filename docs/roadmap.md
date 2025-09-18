# AIDB Performance Roadmap

This roadmap captures the major tracks required to take AIDB from the current scaffold to a best-in-class, next-generation AI database. Each track is split into near-term (weeks), mid-term (quarter), and long-term themes with concrete milestones to guide execution.

## Guiding Principles
- **Tight feedback loops**: land instrumentation, micro-benchmarks, and regression tests alongside new capabilities.
- **Hardware efficiency first**: treat memory bandwidth, cache locality, and SIMD/GPU utilization as first-class constraints.
- **Transactional safety**: marry vector indexing with MVCC so consistency scales together with throughput.
- **Composable surfaces**: expose the same interface (HTTP/gRPC/SQL) regardless of index or storage backend.

## Track 1 – Indexing & Recall Quality
- **Weeks (0-6)**
  - Harden HNSW: visited-set reuse, neighbor pruning heuristics, configurable level-normalization, and background rebuild jobs.
  - SIMD kernels: AVX2/AVX-512 distance paths with runtime feature detection; fall back to scalar for portability.
  - Batch search API: search multiple queries in one pass to amortize memory fetches.
- **Quarter (6-18 weeks)**
  - IVF-Flat with PQ compression (add training loop, centroid persistence, residual encoding).
  - Adaptive index selection: cost model that chooses between BF/HNSW/IVF based on collection density and latency SLOs.
  - Quantization toolkit: OPQ rotation, Product Quantization, scalar quantization for metadata-aware pruning.
- **Longer Term**
  - GPU-resident ANN via cuVS/RAFT bindings and unified planner for hybrid CPU/GPU search.
  - Learned indexes and re-rankers (e.g., ColBERT-lite) for hybrid semantic/text workloads.

## Track 2 – Storage & Durability
- **Weeks**
  - Integrate WAL with page storage: align WAL truncation and snapshot checkpoints with segment lifecycles.
  - Memory-mapped segment files for read-mostly workloads; switch read path to zero-copy slices.
  - Crash-consistency test harness using random fault injection + fsync fuzzing.
- **Quarter**
  - LSM-inspired tiering: hot in-memory segments, warm SSD segments, and compaction scheduling with tombstone GC.
  - MVCC ↔ index hooks so visibility rules drive index updates (xmin/xmax aware search, rollback-safe ingestion).
  - Columnar metadata store (typed columns, bitmap indexes) for predicate pushdown prior to vector scoring.
- **Longer Term**
  - Incremental checkpoints + remote replication targets.
  - Tiered storage (NVMe, S3) with async prefetch & TTL policies.

## Track 3 – Serving & Query Surface
- **Weeks**
  - gRPC service with streaming ingestion and server-side batching knobs (`ef_search`, `nprobe`, batch size).
  - Structured metrics (OpenTelemetry + Prometheus) covering latency histograms and per-collection stats.
  - Query planner skeleton that can fan out to multiple shards/index types.
- **Quarter**
  - SQL surface (Postgres wire compatible subset) translating to vector + metadata operators.
  - AuthN/Z story (API keys, mTLS) and multi-tenant resource limits.
  - Reinforced observability: tracing, structured logging, heatmaps for cache hit/miss.
- **Longer Term**
  - Dynamic rebalancing and shard placement with RAFT/consensus backing.
  - Workload-aware scheduling (NUMA pinning, query routing, GPU/cpu orchestration).

## Track 4 – Benchmarking & Tooling
- **Weeks**
  - Refresh `aidb-bench` with reproducible configs, CPU affinity pinning, and HTML/Markdown report exporters.
  - Micro-benchmarks for kernels (distance, quantization) and integration with Criterion.rs.
  - Automated flamegraph capture scripts (pprof + cargo-flamegraph) for regressions.
- **Quarter**
  - End-to-end soak tests with synthetic + customer workloads, failure injection, and p99 budget alerts.
  - Continuous benchmarking on dedicated hardware with historical trend dashboards.
  - Configurable workload generator (mixture of inserts/updates/deletes/search).
- **Longer Term**
  - Public benchmark harness (ANN-Benchmarks compatible) for marketing-grade comparisons.
  - Chaos-style longevity testing with rolling upgrades, node drains, and index rebuilds under load.

## Track 5 – Developer Experience
- **Weeks**
  - One-command dev stack (`just` recipes) and fixture generators for datasets.
  - Documentation revamp with architecture diagrams, index/storage deep dives, and troubleshooting guides.
  - Swift/macOS admin app feature parity with CLI (collection ops, live metrics, query console).
- **Quarter**
  - In-repo RFC process and ADR templates to capture major design decisions.
  - Workspace linting (clippy profiles, rustfmt guards) and safety rails in CI for unsafe/asm usage.
  - Client SDKs (TypeScript, Python, Go) with load-balanced connection pooling.
- **Longer Term**
  - Self-serve benchmarking portal + hosted sandbox cluster.
  - Schema evolution tooling and migrator for metadata columns.

## Immediate Execution Backlog
1. Land SIMD kernel refactor with proper lane loading and length guards.
2. Add HNSW consistency tests (insert/update/delete) + parameter validation.
3. Wire WAL checkpoints to snapshot endpoints and document operational runbook.
4. Instrument HTTP server with histograms + structured logging.
5. Spin up nightly benchmarks tracking QPS, recall@k, p99 latency per index type.

Each item should close with updated docs, automated tests, and performance notes. This keeps velocity aligned with the "most performant database" objective while maintaining correctness and developer ergonomics.
