"""Process-level settings. Everything per-run lives in `RunConfig` instead."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# src/analyst/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

    cache_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("ANALYST_CACHE_DIR", ".cache")
    )
    runs_db: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("ANALYST_RUNS_DB", "runs/runs.sqlite3")
    )
    out_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("ANALYST_OUT_DIR", "out")
    )
    prompts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "prompts")

    # Politeness. SEC asks for <= 10 req/s; we are far more conservative.
    requests_per_second: float = 3.0
    request_timeout_s: float = 20.0
    max_fetch_concurrency: int = 6

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
