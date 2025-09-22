# Query Statistics and Cost-Based Optimizer Milestones

This document expands the optimizer roadmap from Milestone 0.1 through Milestone 1.10. Each section documents scope, key design artifacts, implementation notes, and validation strategy. The milestones follow two macro phases:

- **Milestone 0.x** establishes a statistics catalog and collection/maintenance primitives that feed the optimizer.
- **Milestone 1.x** builds the memo-based logical optimizer, vectorized execution, and surrounding runtime services that exploit the collected statistics.

> **Missing section notice:** The original outline jumps from Milestone 0.3 to Milestone 1.1. A placeholder is left here because no Milestone 1.0 requirements were provided.

## Milestone 0.1 – Catalog & Statistics Blueprint

**Goal:** Produce a design blueprint for the statistics subsystem that will power future optimizer work.

**Scope:**
- Author the statistics catalog schema: table layout for table-level stats, per-column stats, histograms, and refresh metadata.
- Enumerate histogram variants (equi-width, equi-depth, hybrid sketches) and define where they apply.
- Specify triggers for collecting and updating stats (manual ANALYZE, automatic thresholds, background refresh).
- Define the Rust API surface for reading and writing catalog entries.
- Map integration points with the optimizer (cardinality estimator, cost model, plan validation).

**Deliverables:** Draft design document with schema diagrams, API signatures, and lifecycle diagrams for stats maintenance.

**Validation:** Internal design review and schema review with senior engineers; ensure catalog unit tests cover serialization/deserialization of metadata structures.

## Milestone 0.2 – Basic Statistics Collection

**Goal:** Implement the first ANALYZE flow that captures baseline statistics.

**Scope:**
- Implement an `ANALYZE` command/job that scans relations to compute row counts, number of distinct values (NDV), and min/max per column.
- Persist collected data into the new catalog tables with versioning and timestamps per column.
- Provide a Rust API that exposes strongly typed readers returning versioned statistics and collection timestamps.

**Implementation Notes:**
- Leverage streaming aggregation for NDV (e.g., HyperLogLog) to avoid full materialization.
- Ensure column stats capture null fraction and width estimates needed by the cost model.

**Validation:** Add integration tests that run ANALYZE over synthetic datasets (uniform, skewed, sparse) and assert stored values match expected counts.

## Milestone 0.3 – Histogram Infrastructure

**Goal:** Extend the statistics system with histograms and refresh plumbing.

**Scope:**
- Build equi-width and equi-depth histogram builders with incremental maintenance hooks (merge/split when data drifts).
- Introduce a background task scheduling API that can queue stats refresh jobs (periodic or triggered).
- Record refresh policies (staleness thresholds, row change ratios) in the stats catalog.

**Implementation Notes:**
- Implement histogram serialization formats and compression for catalog storage.
- Provide estimator helpers that return bucket selectivity for predicate types (range, inequality, IN-list).

**Validation:** Add histogram accuracy tests that compare estimated selectivity against observed counts across distributions, and tests covering refresh scheduling/backoff behavior.

## Milestone 1.1 – Logical Algebra & Memo Framework

**Goal:** Establish the logical planning core with memoization and rule infrastructure.

**Scope:**
- Define logical plan nodes for scans, joins, filters, aggregations, projections, vector operators, etc.
- Implement a Cascades-style memo structure with groups, group expressions, and property tracking.
- Create a rewrite rule interface and port the existing heuristics planner into rule form as the baseline rule set.

**Implementation Notes:**
- Ensure memo supports properties required later (distribution, ordering, batch mode) even if unimplemented initially.
- Provide tracing instrumentation to visualize memo growth for debugging.

**Validation:** Plan equivalence tests verifying current SQL queries produce plans identical to the legacy planner; ensure memo invariants hold via unit tests.

## Milestone 1.2 – Cost Model & Cardinality Estimation

**Goal:** Wire statistics into a reusable cost and cardinality estimation subsystem.

**Scope:**
- Implement the cost model that consumes table stats, column stats, and histograms to estimate I/O, CPU, and memory costs.
- Provide a cardinality estimation API with fallbacks when stats are missing or stale.
- Integrate selectivity estimation for conjunction/disjunction and join predicates using histograms and heuristics.

**Implementation Notes:**
- Support configurable calibration parameters (cpu_per_tuple, io_per_page) loaded from configuration.
- Surface estimation confidence intervals for adaptive query execution to leverage later.

**Validation:** Unit tests comparing estimated vs. observed cardinalities on join graphs; regression tests for cost calculations using controlled datasets.

## Milestone 1.3 – Initial Cost-Based Optimizer

**Goal:** Enable the memo + cost model to produce optimal plans with join enumeration.

**Scope:**
- Implement join enumeration (DP-based or memo-driven) and access path selection (index vs. seq scan).
- Integrate the optimizer into the execution pipeline with graceful fallback to heuristic plans on failure/timeouts.
- Ensure cost model feedback influences rule application ordering.

**Implementation Notes:**
- Add timeout guards and memo pruning to prevent explosion on complex queries.
- Emit optimizer diagnostics (chosen plan, rejected alternatives) for observability.

**Validation:** Dedicated optimizer test suite plus subset of TPC-H queries comparing performance and plan quality to baseline.

## Milestone 1.4 – Vectorized Execution Core

**Goal:** Introduce columnar batch processing to accelerate execution.

**Scope:**
- Design a columnar batch format with alignment for SIMD and GPU operators.
- Build scan/filter/projection operators that operate over columnar batches while supporting fallback row mode.
- Introduce batch-aware plan nodes and an executor driver capable of switching between batch and row operators.

**Implementation Notes:**
- Ensure memory management integrates with buffer pool and respects lifetimes for zero-copy handoff.
- Provide instrumentation for batch sizes, operator throughput, and fallback triggers.

**Validation:** Microbenchmarks comparing vectorized vs row-at-a-time operators, plus correctness tests that cover mixed-mode pipelines.

## Milestone 1.5 – Parallel Execution Framework

**Goal:** Scale execution through intra-query parallelism.

**Scope:**
- Build a task scheduler with work-stealing queues and cooperative cancellation.
- Implement parallel scans and hash joins operating on batches, respecting partitioning requirements.
- Expose configuration knobs for degree of parallelism at session and system level.

**Implementation Notes:**
- Integrate scheduler metrics (queue depth, worker utilization) for monitoring.
- Ensure vectorized operators cooperate with scheduler by chunking work into balanced tasks.

**Validation:** Stress tests on large datasets and concurrency correctness checks (race detection, deterministic outcomes with MVCC visibility rules).

## Milestone 1.6 – Adaptive Query Execution

**Goal:** Adjust plans at runtime using observed statistics.

**Scope:**
- Capture runtime stats from operators (row counts, skew metrics) and feed them into an adaptive controller.
- Implement plan adjustments such as switching join strategies, repartitioning, or changing parallelism mid-query.
- Update catalog or session-level stats with feedback loops where safe.

**Implementation Notes:**
- Define thresholds and hysteresis to avoid thrashing when adaptivity triggers.
- Provide visibility into adaptive decisions via EXPLAIN ANALYZE.

**Validation:** Scenario tests that force adaptive behavior (skewed joins, misestimated predicates) and ensure convergence/correctness compared to static plans.

## Milestone 1.7 – JIT Compilation for Hot Operators

**Goal:** Accelerate critical operators through JIT compilation.

**Scope:**
- Integrate Cranelift/LLVM backend for generating filter and projection kernels from expression trees.
- Add lightweight profiling to identify hot loops eligible for JIT.
- Manage generated code cache with invalidation hooks tied to schema changes.

**Implementation Notes:**
- Provide sandboxing and guardrails for JIT (code size limits, fallback on compilation failure).
- Support mixing JIT and interpreted operators in a single pipeline.

**Validation:** Performance benchmarks comparing JIT vs interpreted execution, ensuring correctness parity and measuring compilation overhead.

## Milestone 1.8 – Common Subexpression & Result Cache

**Goal:** Avoid recomputation through caching at plan and operator level.

**Scope:**
- Implement normalized plan/parameter signatures for identifying reusable work.
- Cache evaluation results and intermediate batch outputs with memory/accounting policies.
- Add invalidation hooks linked to data modifications and materialized view updates.

**Implementation Notes:**
- Provide cache introspection APIs (hit/miss counters, memory usage, eviction reasons).
- Integrate with adaptive execution to warm caches on anticipated reuse.

**Validation:** Workloads demonstrating cache hits and correct invalidation when underlying data changes.

## Milestone 1.9 – Materialized Views Framework

**Goal:** Introduce materialized view definitions and optimizer rewrites.

**Scope:**
- Extend catalog to store MV definitions, refresh strategies (sync/async), and dependencies.
- Implement refresh executors (full refresh initially) and scheduling integration.
- Add optimizer rules that recognize eligible query patterns and rewrite to use MVs.

**Implementation Notes:**
- Ensure MV refresh interacts cleanly with transaction boundaries and stats catalog (trigger re-analyze when MV updates).
- Provide EXPLAIN annotations showing MV usage.

**Validation:** MV correctness tests, refresh scheduling coverage, and rewrite tests verifying plan substitution when valid.

## Milestone 1.10 – Query Federation Skeleton

**Goal:** Lay the groundwork for federated query capabilities.

**Scope:**
- Define external source abstraction and connector API; implement a reference connector (e.g., Postgres or Parquet).
- Extend cost model with source-aware costing and pushdown rules for predicates/projections.
- Support cross-source plan fragments with data movement operators.

**Implementation Notes:**
- Ensure connector lifecycle integrates with catalog (capabilities, stats refresh hooks).
- Provide security/configuration model for external credentials.

**Validation:** End-to-end federated query tests combining local and external sources, covering pushdowns, fallback paths, and failure handling.

