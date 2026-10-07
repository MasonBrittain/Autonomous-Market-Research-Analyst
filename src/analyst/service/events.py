"""Per-run progress events, persisted so another process can stream them.

The worker and the API are separate processes -- in production, separate
containers -- so progress cannot travel through memory. Every pipeline progress
callback and every worker lifecycle change (claimed, retry scheduled, lease lost)
is appended here, and the API's event stream tails the table.

Events are append-only with a monotonically increasing id, which is exactly the
shape Server-Sent Events wants: the id goes out as the SSE `id:` field, so a
browser that drops its connection reconnects with `Last-Event-ID` and resumes
where it left off instead of replaying or missing anything.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from ..db import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS run_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  TEXT NOT NULL,
    ts      REAL NOT NULL,
    source  TEXT NOT NULL,
    node    TEXT,
    status  TEXT NOT NULL,
    detail  TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, id);
"""

# Display names for pipeline nodes, shared by the event stream and the CLI.
NODE_LABELS = {
    "resolve": "Resolve",
    "scout": "Scout",
    "curate": "Librarian",
    "analyze": "Analyst",
    "challenge": "Adversary",
    "compose": "Scribe",
}


@dataclass(frozen=True)
class Event:
    id: int
    run_id: str
    ts: float
    source: str
    node: str | None
    status: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "ts": self.ts,
            "source": self.source,
            "node": self.node,
            "label": NODE_LABELS.get(self.node or "", self.node),
            "status": self.status,
            "detail": self.detail,
        }


class EventLog:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = path
        self.clock = clock
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return connect(self.path)

    def append(
        self,
        run_id: str,
        status: str,
        *,
        node: str | None = None,
        detail: str = "",
        source: str = "pipeline",
    ) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """INSERT INTO run_events (run_id, ts, source, node, status, detail)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, self.clock(), source, node, status, detail[:1000]),
            )
            return int(cursor.lastrowid or 0)

    def since(self, run_id: str, after_id: int = 0, *, limit: int = 500) -> list[Event]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT * FROM run_events WHERE run_id = ? AND id > ?
                   ORDER BY id LIMIT ?""",
                (run_id, after_id, limit),
            ).fetchall()
        return [
            Event(
                id=r["id"],
                run_id=r["run_id"],
                ts=r["ts"],
                source=r["source"],
                node=r["node"],
                status=r["status"],
                detail=r["detail"],
            )
            for r in rows
        ]

    def recorder(self, run_id: str) -> Callable[[str, str, str], None]:
        """A pipeline progress callback that persists each update."""

        def record(node: str, status: str, detail: str) -> None:
            self.append(run_id, status, node=node, detail=detail, source="pipeline")

        return record
