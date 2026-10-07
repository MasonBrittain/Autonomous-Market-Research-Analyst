"""What counts as "the same research", decided before anything is queued.

Two requests are the same work when they resolve to the same entity and would run
under the same configuration. Comparing raw query strings would treat "AAPL" and
"Apple Inc." as different -- two paid runs for one company -- so the API resolves
the query against the SEC index first and keys on the result.

Resolving up front has a second payoff: an ambiguous query ("Delta") is rejected
inside the request, with the candidates attached, instead of becoming a job that
fails minutes later with nobody watching.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable

from ..models import Entity, RunConfig
from ..prompts import prompt_fingerprint
from ..tools import entity as entity_tools
from ..tools.fetch import Fetcher


def entity_key(entity: Entity) -> str:
    if entity.is_industry:
        return f"industry:{entity_tools.normalize_name(entity.name)}"
    if entity.ticker:
        return f"ticker:{entity.ticker.upper()}"
    return f"cik:{entity.cik or entity_tools.normalize_name(entity.name)}"


def config_fingerprint(config: RunConfig) -> str:
    """Hash of everything that changes a run's output.

    Includes the prompt text: a brief produced before a prompt edit must not be
    served as if it came from the new prompts. Stub runs fingerprint differently
    from live ones for the same reason -- a placeholder brief must never satisfy a
    request for a real one.
    """
    digest = hashlib.blake2b(digest_size=6)
    digest.update(config.model_dump_json().encode())
    digest.update(prompt_fingerprint(config.prompt_version).encode())
    return digest.hexdigest()


def dedupe_key(entity: Entity, config: RunConfig) -> str:
    return f"{entity_key(entity)}@{config_fingerprint(config)}"


class EntityResolver:
    """Resolves queries against the SEC ticker index, held in memory.

    The index is ~800 KB of JSON and ~12k rows; rebuilding it per request would
    dominate request latency. It is cached for `ttl_s` and refreshed afterwards,
    since the SEC adds and delists issuers daily.
    """

    def __init__(
        self,
        fetcher_factory: Callable[[], Fetcher],
        *,
        ttl_s: float = 24 * 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetcher_factory = fetcher_factory
        self._ttl_s = ttl_s
        self._clock = clock
        self._index: entity_tools.CompanyIndex | None = None
        self._loaded_at = 0.0
        self._lock = asyncio.Lock()

    async def index(self) -> entity_tools.CompanyIndex:
        async with self._lock:
            stale = self._clock() - self._loaded_at > self._ttl_s
            if self._index is None or stale:
                self._index = await entity_tools.load_index(self._fetcher_factory())
                self._loaded_at = self._clock()
            return self._index

    async def __call__(self, query: str) -> Entity:
        return entity_tools.resolve_from_index(query, await self.index())
