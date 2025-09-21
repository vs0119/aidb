# AIDB — AI-Optimized Vector Database (Rust)

AIDB is an experimental, high-performance vector database for AI workloads. It
combines fast approximate nearest-neighbor search, durable storage, and a simple
Rust API/HTTP surface so you can prototype retrieval-augmented systems without
wrestling with infrastructure.

## Project Status

> **Alpha** – the architecture is stabilizing, but APIs may change. Follow the
> roadmap below to see what is coming next.

## Features

- Vector similarity search (cosine, L2) with optional metadata filters
- Durable writes with a write-ahead log and in-memory query path
- Modular indexing layer (brute-force today, adaptive/HNSW indexes in progress)
- Axum-based HTTP server for CRUD and search
- Embedded SQL surface for metadata prototyping

## Quick Start

```bash
# Install the Rust toolchain (1.70+) if you have not already.
curl https://sh.rustup.rs -sSf | sh

# Clone and build
git clone https://github.com/aidb-dev/aidb.git
cd aidb
cargo run -p aidb-server
```

The server listens on `0.0.0.0:8080` by default. You can exercise the API with
`curl` or the examples in [`docs/http-api.md`](docs/http-api.md).

### Examples

```bash
# Run the crate-level example that adds and queries vectors
cargo run --package aidb-core --example basic
```

### Testing & Benchmarks

```bash
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo bench --workspace --no-run
```

These commands are enforced in CI on pull requests and pushes to `main`.

## Documentation

- [Core crate docs](crates/aidb-core/src/lib.rs)
- [HTTP API](docs/http-api.md)
- [SQL engine](docs/sql.md)
- [Bench harness](apps/README.md)
- [Contributing](CONTRIBUTING.md)

Generated documentation is available locally via `cargo doc --open`.

## Release Process

We follow [Semantic Versioning](https://semver.org/). Release tags take the form
`vMAJOR.MINOR.PATCH`. The release workflow:

1. Update [`CHANGELOG.md`](CHANGELOG.md) with highlights for the version.
2. Ensure `main` is green (fmt, clippy, test, benchmarks).
3. Tag the commit (`git tag vX.Y.Z && git push origin vX.Y.Z`).
4. Let the "Release" GitHub Actions workflow build artifacts and publish notes.

## Roadmap

| Quarter | Focus | Highlights |
| ------- | ----- | ---------- |
| 2024 Q2 | Core fidelity | SIMD kernels, correctness checks, WAL durability |
| 2024 Q3 | Indexing | HNSW graph search, IVF-Flat, quantization experiments |
| 2024 Q4 | Serving | Horizontal sharding, observability, SQL planner |
| 2025 Q1 | Ecosystem | Client SDKs, dashboard, managed hosting preview |

We track issues with the `roadmap` label; contributions welcome!

## Community & Governance

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Contributing Guide](CONTRIBUTING.md)
- Issue templates auto-label bug triage (`bug`, `enhancement`, `good first issue`)
- CODEOWNERS ensure domain experts review changes to critical components

## License

AIDB is distributed under the terms of the [MIT License](LICENSE).

## Acknowledgements

This project stands on the shoulders of the Rust community and the fantastic
vector search research ecosystem. Thank you for testing, profiling, and sharing
feedback!
