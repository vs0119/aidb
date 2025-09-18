# AIDB Mac App (SwiftUI)

A lightweight native macOS UI for administering and querying the AIDB server.

## Features
- Configure server base URL and check health
- List collections and view stats
- Create collection (HNSW or brute‑force) with parameters
- Upsert points (single or batch JSON)
- Run vector search with optional metadata filter
- Admin actions: snapshot, compact, update HNSW ef_search

## Build & Run
- Open `apps/aidb-mac/Package.swift` in Xcode and run the `AIDBMac` scheme.
- Or from CLI: `swift run --package-path apps/aidb-mac AIDBMac`
- Ensure the Rust server is running: `cargo run -p aidb-server`

## Notes
- The executable target runs as a SwiftUI app on macOS 13+ when launched via Xcode.
- This is an MVP; polish, error states, and design can be improved.
