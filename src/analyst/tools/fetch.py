"""The one place in the codebase that touches the network.

Every adapter goes through `Fetcher`, which gives us four things in one spot:
the disk cache, per-host rate limiting, robots.txt compliance, and a descriptive
User-Agent. Centralising it means politeness cannot be accidentally bypassed by
a new adapter, and it means the whole pipeline can be run offline from cache.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from ..cache import FetchCache
from ..config import Settings, settings


@dataclass
class FetchResult:
    url: str
    status: int
    body: str
    from_cache: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300 and bool(self.body)


class Fetcher:
    def __init__(
        self,
        cache: FetchCache | None = None,
        config: Settings | None = None,
        *,
        offline: bool = False,
    ) -> None:
        self.cfg = config or settings()
        self.cache = cache if cache is not None else FetchCache()
        # offline=True makes cache misses fail instead of hitting the network --
        # used by tests and by eval replays that must not vary.
        self.offline = offline
        self._robots: dict[str, RobotFileParser | None] = {}
        self._last_request: dict[str, float] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._sem = asyncio.Semaphore(self.cfg.max_fetch_concurrency)

    # -- politeness -------------------------------------------------------- #

    def _host(self, url: str) -> str:
        return urlparse(url).netloc.lower()

    def _lock(self, host: str) -> asyncio.Lock:
        if host not in self._host_locks:
            self._host_locks[host] = asyncio.Lock()
        return self._host_locks[host]

    async def _throttle(self, host: str) -> None:
        """Serialise per host and enforce a minimum gap between requests."""
        min_gap = 1.0 / max(self.cfg.requests_per_second, 0.1)
        async with self._lock(host):
            last = self._last_request.get(host)
            if last is not None:
                wait = min_gap - (time.monotonic() - last)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_request[host] = time.monotonic()

    async def _robots_allows(self, client: httpx.AsyncClient, url: str) -> bool:
        host = self._host(url)
        if host not in self._robots:
            parser: RobotFileParser | None = None
            robots_url = f"{urlparse(url).scheme}://{host}/robots.txt"
            try:
                resp = await client.get(robots_url, timeout=8.0)
                if resp.status_code == 200:
                    parser = RobotFileParser()
                    parser.parse(resp.text.splitlines())
            except (httpx.HTTPError, UnicodeDecodeError):
                parser = None  # Unreachable robots.txt is treated as permissive.
            self._robots[host] = parser
        parser = self._robots[host]
        if parser is None:
            return True
        return parser.can_fetch(self.cfg.sec_user_agent, url)

    # -- fetching ---------------------------------------------------------- #

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
        check_robots: bool = True,
    ) -> FetchResult:
        if use_cache:
            cached = self.cache.get(url)
            if cached is not None:
                return FetchResult(
                    url=url,
                    status=int(cached.get("status", 200)),
                    body=str(cached.get("body", "")),
                    from_cache=True,
                )

        if self.offline:
            return FetchResult(url=url, status=0, body="", error="offline: cache miss")

        request_headers = {
            "User-Agent": self.cfg.sec_user_agent,
            # SEC's access guidelines ask for this explicitly.
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "en-US,en;q=0.9",
            **(headers or {}),
        }

        async with self._sem:
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=self.cfg.request_timeout_s
                ) as client:
                    if check_robots and not await self._robots_allows(client, url):
                        return FetchResult(
                            url=url, status=0, body="", error="blocked by robots.txt"
                        )
                    await self._throttle(self._host(url))
                    resp = await client.get(url, headers=request_headers)
                    body = resp.text
            except httpx.HTTPError as exc:
                return FetchResult(url=url, status=0, body="", error=f"{type(exc).__name__}: {exc}")

        if 200 <= resp.status_code < 300 and body:
            self.cache.put(url, body, resp.status_code)
        if resp.status_code == 403 and "sec.gov" in self._host(url):
            # Almost always the User-Agent, not an IP block -- say so.
            return FetchResult(
                url=url,
                status=403,
                body="",
                error=(
                    "sec.gov returned 403; this is nearly always a User-Agent without a "
                    "contact email. Set SEC_USER_AGENT in .env."
                ),
            )
        return FetchResult(url=url, status=resp.status_code, body=body)

    async def get_many(
        self, urls: list[str], *, headers: dict[str, str] | None = None, check_robots: bool = True
    ) -> list[FetchResult]:
        tasks = [self.get(u, headers=headers, check_robots=check_robots) for u in urls]
        return list(await asyncio.gather(*tasks))

    def stats(self) -> dict[str, object]:
        return {"cache": self.cache.stats(), "hosts_seen": len(self._last_request)}
