"""Content-addressed disk cache for fetched documents.

This is load-bearing for development speed, not just politeness: with every HTTP
response cached by URL, the Librarian / Analyst / Adversary stages can be
re-run and re-tuned hundreds of times against a frozen evidence set without a
single network request. It is also what makes eval runs comparable -- two runs
over the same cache differ only by model behaviour.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .config import settings


def _key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class FetchCache:
    def __init__(self, root: Path | None = None, ttl_seconds: int | None = None) -> None:
        self.root = root or settings().cache_dir
        # News bodies are immutable once published; a long TTL is correct here.
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, url: str) -> Path:
        k = _key(url)
        # Shard by first two chars so directories stay browsable.
        d = self.root / k[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{k}.json"

    def get(self, url: str) -> dict[str, Any] | None:
        p = self._path(url)
        if not p.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.misses += 1
            return None
        if self.ttl_seconds is not None:
            age = time.time() - payload.get("cached_at", 0)
            if age > self.ttl_seconds:
                self.misses += 1
                return None
        self.hits += 1
        return payload

    def put(
        self, url: str, body: str, status: int = 200, meta: dict[str, Any] | None = None
    ) -> None:
        payload = {
            "url": url,
            "status": status,
            "body": body,
            "meta": meta or {},
            "cached_at": time.time(),
        }
        tmp = self._path(url).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._path(url))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else 0.0

    def stats(self) -> dict[str, object]:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": self.hit_rate}
