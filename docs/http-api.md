# HTTP API Quick Reference

AIDB exposes a JSON/HTTP API via the `aidb-server` crate. The default listener is
`0.0.0.0:8080`.

## Collections

- `POST /collections`
  - Body: `{ "name": "docs", "dim": 768, "metric": "cosine", "wal_dir": "./data" }`
- `GET /collections`
  - Returns the list of configured collections.

## Points

- `POST /collections/{name}/points`
  - Upsert a vector and optional payload.
- `DELETE /collections/{name}/points/{id}`
  - Remove a vector.

## Search

- `POST /collections/{name}/search`
  - Body: `{ "vector": [0.1, 0.2], "top_k": 10, "filter": {"source": "docs"} }`
- `POST /collections/{name}/search:batch`
  - Body: `{ "queries": [[0.1, 0.2], [0.3, 0.4]], "top_k": 10 }`

See [`apps/README.md`](../apps/README.md) for benchmarking helpers and additional
usage examples.
