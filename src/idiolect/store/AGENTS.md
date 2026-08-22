# Store Guide

These instructions apply to `src/idiolect/store/`.

- Keep storage protocols in `base.py` free of DuckDB, SQL, filesystem, and
  backend lifecycle behavior. Consumers depend on the protocols, not concrete
  adapters.
- Keep DuckDB schema and transaction behavior in `duck.py`. Save one source
  event and its normalized records atomically, preserve duplicate-event
  idempotence, and return records in the ordering promised by the port.
- Keep revision replacement, deletion, reaction, mention, quote, and attachment
  relationships consistent when normalized records are updated.
- Do not expose DuckDB rows, connections, or SQL details through public storage
  contracts. Convert them to shared typed records at the adapter boundary.
- Treat database and artifact paths as private runtime state. Tests use temporary
  databases and must create all schema and data they need.
- Keep reserved adapters explicit. Do not add placeholder behavior to a protocol
  or to `files.py` before a real storage contract exists.
