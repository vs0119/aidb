# Contributing to AIDB

Thanks for your interest in improving AIDB! This guide explains how to set up a
local development environment, propose changes, and collaborate with the core
team.

## Code of Conduct

By participating you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
Please report unacceptable behavior to [opensource@aidb.dev](mailto:opensource@aidb.dev).

## Development Workflow

1. Fork the repository and create a feature branch from `main`.
2. Install the Rust toolchain version declared in `rust-toolchain.toml` or the
   `rust-version` in `Cargo.toml`.
3. Run the quality gates before opening a pull request:
   - `cargo fmt -- --check`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `cargo test --workspace --all-features`
   - `cargo bench --workspace --no-run`
4. Push your branch and open a pull request describing your changes and test
   coverage. Pull requests automatically run the same checks in CI.
5. Please keep pull requests focused. If you plan a larger change, open a
   discussion issue first so we can align on design and scope.

## Commit Guidelines

* Keep commits small and logically scoped. Use descriptive commit messages in
the imperative mood (e.g., "Add HTTP error responses").
* Format code with `cargo fmt` and resolve clippy warnings before committing.
* Update documentation, examples, and tests alongside code changes.

## Testing

We expect new features to be covered by unit and/or integration tests. Run the
full workspace test suite before submitting:

```bash
cargo test --workspace --all-features
```

If you add benchmarks or performance-sensitive code, include reproducible steps
in the pull request so reviewers can validate the results.

## Release Process

We follow semantic versioning (`MAJOR.MINOR.PATCH`). Releases are created from
`main` after:

1. Ensuring the CI pipeline is green.
2. Updating `CHANGELOG.md` with highlights for the release.
3. Tagging the commit (`git tag vX.Y.Z && git push origin vX.Y.Z`).
4. Allowing the release workflow to publish release notes and artifacts.

## Questions & Support

If you have questions about architecture or need help getting started, open a
[GitHub discussion](https://github.com/aidb-dev/aidb/discussions) or join our
community chat on Matrix (`#aidb:matrix.org`).

Thanks again for helping us build a reliable, high-performance vector database!
