"""Process-level settings. Everything per-run lives in `RunConfig` instead."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent
# src/analyst/config.py -> repository root. Only meaningful in a source checkout.
PROJECT_ROOT = PACKAGE_DIR.parents[1]


def _data_root() -> Path:
    """Where caches, the run database and reports are written.

    `ANALYST_DATA_DIR` wins when set -- that is how a container points everything
    at one mounted volume. Otherwise a source checkout writes inside the repo (the
    existing developer layout), and an installed package writes to the working
    directory. Anchoring to `__file__` unconditionally, as this used to, would put
    an installed package's data inside site-packages.
    """
    configured = os.getenv("ANALYST_DATA_DIR")
    if configured:
        return Path(configured)
    if (PROJECT_ROOT / "pyproject.toml").exists():
        return PROJECT_ROOT
    return Path.cwd()


def _under_data_root(env_var: str, default: str) -> Path:
    # An absolute value in the env var wins, because `Path("/a") / "/b"` is "/b".
    return _data_root() / os.getenv(env_var, default)


@dataclass(frozen=True)
class Settings:
    api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None)
    model: str = field(default_factory=lambda: os.getenv("ANALYST_MODEL", "claude-opus-5"))

    # SEC blocks requests without a descriptive User-Agent carrying contact info.
    # https://www.sec.gov/os/accessing-edgar-data
    sec_user_agent: str = field(
        default_factory=lambda: os.getenv(
            "SEC_USER_AGENT", "autonomous-market-analyst/0.1 (contact@example.com)"
        )
    )

    cache_dir: Path = field(default_factory=lambda: _under_data_root("ANALYST_CACHE_DIR", ".cache"))
    runs_db: Path = field(
        default_factory=lambda: _under_data_root("ANALYST_RUNS_DB", "runs/runs.sqlite3")
    )
    out_dir: Path = field(default_factory=lambda: _under_data_root("ANALYST_OUT_DIR", "out"))

    # Prompts ship inside the package. They used to live at the repository root,
    # which only resolved from a source checkout: an installed wheel -- and so any
    # container image -- could not find a single prompt.
    prompts_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ANALYST_PROMPTS_DIR") or PACKAGE_DIR / "prompts")
    )

    # Politeness. SEC asks for <= 10 req/s; we are far more conservative.
    requests_per_second: float = 3.0
    request_timeout_s: float = 20.0
    max_fetch_concurrency: int = 6

    # -- service ----------------------------------------------------------- #

    # Bearer token required on endpoints that start paid work. Unset means open,
    # which is right for local use and wrong for anything reachable from outside:
    # every accepted request spends model credits.
    service_api_key: str | None = field(
        default_factory=lambda: os.getenv("ANALYST_SERVICE_API_KEY") or None
    )
    # Comma-separated origins allowed to call the API from a browser.
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in os.getenv("ANALYST_CORS_ORIGINS", "").split(",") if o.strip()
        )
    )
    # A completed brief younger than this is served again instead of re-run.
    report_ttl_s: float = field(
        default_factory=lambda: float(os.getenv("ANALYST_REPORT_TTL_SECONDS", str(24 * 3600)))
    )
    # How long a worker owns a job before another may take it over, and how often
    # it renews that claim. The ratio matters more than either number: the lease
    # must survive several missed heartbeats.
    job_lease_s: float = field(
        default_factory=lambda: float(os.getenv("ANALYST_JOB_LEASE_SECONDS", "120"))
    )
    job_heartbeat_s: float = field(
        default_factory=lambda: float(os.getenv("ANALYST_JOB_HEARTBEAT_SECONDS", "20"))
    )
    job_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("ANALYST_JOB_MAX_ATTEMPTS", "3"))
    )

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    @property
    def sec_user_agent_valid(self) -> bool:
        """SEC rejects any request whose User-Agent lacks a contact email.

        Verified against the live endpoint: a descriptive UA without an address
        returns 403, the same UA with one returns 200. Checking up front turns a
        confusing mid-pipeline failure into an actionable startup message.
        """
        return bool(re.search(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", self.sec_user_agent, re.I))

    def sec_user_agent_problem(self) -> str | None:
        if self.sec_user_agent_valid:
            return None
        return (
            "SEC_USER_AGENT must contain a contact email address or sec.gov will "
            "return 403 on every request. Set it in .env, for example:\n"
            '  SEC_USER_AGENT="autonomous-market-analyst/0.1 (you@example.com)"'
        )

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.runs_db.parent.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()
