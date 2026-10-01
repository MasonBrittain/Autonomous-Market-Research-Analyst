"""SQLite checkpoint store.

The whole run is serialized after every node. That buys two things: a killed run
resumes without re-fetching or re-paying for completed stages, and every finished
run is a durable artifact an eval harness can re-score without re-running it.

SQLite now, Azure SQL later -- the interface is deliberately narrow (save, load,
list, delete) so swapping the backend touches only this file. The production
queue will use `UPDATE ... WITH (UPDLOCK, READPAST) ... OUTPUT`, the T-SQL
equivalent of SKIP LOCKED, against the same row shape.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from ..config import settings
from ..models import ResearchRun

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    entity_name TEXT,
    ticker      TEXT,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    cost_usd    REAL NOT NULL DEFAULT 0,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_ticker ON runs(ticker);
CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_at DESC);
"""


@dataclass
class RunSummary:
    id: str
    query: str
    entity_name: str | None
    ticker: str | None
    status: str
    updated_at: str
    cost_usd: float


class RunStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings().runs_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, run: ResearchRun) -> None:
        run.touch()
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO runs (id, query, entity_name, ticker, status, created_at,
                                  updated_at, cost_usd, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    entity_name = excluded.entity_name,
                    ticker      = excluded.ticker,
                    status      = excluded.status,
                    updated_at  = excluded.updated_at,
                    cost_usd    = excluded.cost_usd,
                    payload     = excluded.payload
                """,
                (
                    run.id,
                    run.query,
                    run.entity.name if run.entity else None,
                    run.entity.ticker if run.entity else None,
                    run.status.value,
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                    run.ledger.total_usd,
                    run.model_dump_json(),
                ),
            )
            conn.commit()

    def load(self, run_id: str) -> ResearchRun | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT payload FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return ResearchRun.model_validate_json(row["payload"])

    def latest_for(self, query: str) -> ResearchRun | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT payload FROM runs WHERE query = ? ORDER BY updated_at DESC LIMIT 1",
                (query,),
            ).fetchone()
        return ResearchRun.model_validate_json(row["payload"]) if row else None

    def list_runs(self, limit: int = 20) -> list[RunSummary]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT id, query, entity_name, ticker, status, updated_at, cost_usd
                   FROM runs ORDER BY updated_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            RunSummary(
                id=r["id"],
                query=r["query"],
                entity_name=r["entity_name"],
                ticker=r["ticker"],
                status=r["status"],
                updated_at=r["updated_at"],
                cost_usd=r["cost_usd"] or 0.0,
            )
            for r in rows
        ]

    def delete(self, run_id: str) -> bool:
        with closing(self._connect()) as conn:
            cursor = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0
