# AIDB Operational Runbook

## Snapshot & WAL lifecycle

A collection snapshot now acts as the authoritative on-disk checkpoint:

1. When `Collection::snapshot_to` runs (or the `/collections/:name/snapshot` API is called), we serialize the in-memory index and **truncate the collection WAL back to zero bytes** under the same lock used for writes.
2. Subsequent mutations append to a fresh WAL buffer. On restart, restore by loading the snapshot first and then replaying the remaining WAL to pick up the delta since that checkpoint.
3. If you manually invoke `load_snapshot`, the WAL is left intact—allowing you to replay newer operations if desired. If you intend to discard those ops, call `snapshot_to` immediately after `load_snapshot` to clear the WAL from the new baseline.

### Recommended workflow

- **Taking a checkpoint**
  - `POST /collections/:name/snapshot` → copies the index to your target path and zeroes the WAL.
  - Copy the snapshot file to durable storage.
- **Restoring**
  1. Create a fresh collection pointing at the same WAL path.
  2. Call `load_snapshot` with the saved snapshot file.
  3. Call `recover` to replay any WAL entries that were generated after the checkpoint.
  4. Optionally call `snapshot_to` to seal the restored state as the new baseline.

The WAL and snapshot coordination now runs under a shared mutex, ensuring no updates are lost while the snapshot is being written.

## Monitoring expectations

- `snapshot_to` and `load_snapshot` hold the WAL mutex for the duration of their work. Heavy write workloads may observe a brief stall while the checkpoint is written.
- WAL truncation uses `File::set_len(0)` followed by `sync_all`, so the filesystem must support truncation; on failure, the snapshot call will bubble up an I/O error.
- Every collection exposes `wal_size_bytes` and `wal_last_truncate` via the collection info APIs (`GET /collections` and `GET /collections/:name`). You can alert when `wal_size_bytes` exceeds your retention budget or when `wal_last_truncate` is `null` for too long (meaning no snapshot has executed since process start).
- Storage logs emit `aidb::storage` entries named `wal_growth` (every 64 MiB increment) and `wal_truncated`. Feed these into your log aggregator to build dashboards without waiting for a full metrics pipeline.
- WAL recovery auto-switches to a memory-mapped read path for large logs (default ≥1 MiB, 1 KiB in tests), so restarting after heavy ingest stays O(bytes) without extra copies.
- `/metrics` now serves Prometheus-style gauges: `aidb_wal_size_bytes{collection="..."}` and `aidb_wal_seconds_since_truncate{collection="..."}`. Plug this into your scraper to watch WAL drift over time.

## Future extensions

- Add WAL file rotation metrics so operators can alert if a snapshot has not been taken recently.
- Expose admin APIs to force WAL truncation or to enumerate checkpoints for cloud backups.
