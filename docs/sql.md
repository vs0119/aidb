# SQL-style Command Interface

AIDB exposes a `/sql` endpoint that accepts a compact SQL-inspired syntax for
managing collections and running vector searches without switching to the JSON
HTTP APIs.

## Supported statements

### Create a collection

```
CREATE COLLECTION docs (
  DIM = 768,
  METRIC = 'cosine',
  INDEX = 'hnsw',
  WAL_DIR = './data',
  M = 16,
  EF_CONSTRUCTION = 200,
  EF_SEARCH = 64
);
```

Required fields:

- `DIM` – vector dimensionality.

Optional fields:

- `METRIC` – `cosine` (default) or `euclidean`.
- `INDEX` – `hnsw` (default) or `bruteforce`.
- `WAL_DIR` – directory for the write-ahead log.
- `M`, `EF_CONSTRUCTION`, `EF_SEARCH` – HNSW tuning knobs applied when the
  index type is `hnsw`.

### Insert a vector

```
INSERT INTO docs VALUES (
  ID = '550e8400-e29b-41d4-a716-446655440000',
  VECTOR = [0.1, 0.2, 0.3, 0.4],
  PAYLOAD = {"source": "a.txt"}
);
```

- `ID` is optional – a UUID will be generated when omitted.
- `VECTOR` is required and must be a JSON array of floats.
- `PAYLOAD` is optional JSON metadata that is stored alongside the vector.

### Search a collection

```
SEARCH docs (
  VECTOR = [0.1, 0.2, 0.3, 0.4],
  TOPK = 5,
  FILTER = {"source": "a.txt"}
);
```

- `VECTOR` is required and must match the collection dimensionality.
- `TOPK` is required and controls the number of nearest neighbours returned.
- `FILTER` is optional JSON matching the exact-metadata filter used in the REST
  API (`equals` semantics on the provided key/value pairs).

## Responses

The endpoint returns a discriminated JSON payload:

- `{"type": "create", "ok": true}` for `CREATE COLLECTION`.
- `{"type": "insert", "id": "..."}` for `INSERT`.
- `{"type": "search", "results": [...]}` for `SEARCH`, reusing the existing
  REST search result schema (`id`, `score`, `payload`).

Errors are returned with HTTP 400 when the SQL is invalid or with HTTP 404/500
when execution fails.
