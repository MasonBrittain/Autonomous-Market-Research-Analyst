"""SQLite connections shared by the run store, the job queue and the event log.

All three live in one database file on purpose. Two guarantees depend on it:

* Enqueueing a job and creating its run happen in one transaction, so a crash
  cannot leave a job pointing at a run that does not exist.
* A worker's checkpoint write is fenced against its job lease in the same
  statement (`UPDATE runs ... WHERE EXISTS (SELECT 1 FROM jobs ...)`), so a worker
  that has lost its lease cannot overwrite the worker that replaced it.

Neither would be possible across two databases without distributed transactions.
The same holds when this moves to Azure SQL: one database, same transactions.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Generous, because a writer waiting on a lock is normal under load and a
# "database is locked" error is not. A checkpoint of a large run takes a few ms.
BUSY_TIMEOUT_S = 30.0


def connect(path: Path, *, timeout: float = BUSY_TIMEOUT_S) -> sqlite3.Connection:
    """Open a connection in autocommit mode with WAL enabled.

    Autocommit (`isolation_level=None`) makes every single statement its own
    transaction and leaves multi-statement transactions explicit -- see
    `transaction()`. Python's default instead opens transactions implicitly on the
    first write, which makes it easy to hold a write lock longer than intended.

    WAL lets readers proceed while a writer is active, which is what allows the
    API to serve status requests while a worker checkpoints a run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=timeout, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """`BEGIN IMMEDIATE` ... `COMMIT`, rolling back on any exception.

    IMMEDIATE takes the write lock up front rather than on first write. That is
    what makes a read-then-write sequence inside the block atomic with respect to
    other writers: nobody can slip in between the check and the insert.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
