"""Prompt loader.

Prompts live in `analyst/prompts/<version>/*.md` rather than inline string literals
so a run can record which prompt revision produced it. Without that, an eval score
is unattributable -- you cannot tell whether a regression came from a prompt edit,
a code change, or the model.

They are package data, shipped inside the wheel, because the pipeline cannot run
without them. `ANALYST_PROMPTS_DIR` overrides the location for prompt experiments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..config import settings


class PromptNotFound(FileNotFoundError):
    pass


@lru_cache(maxsize=64)
def load_prompt(name: str, version: str = "v1") -> str:
    path = settings().prompts_dir / version / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in (settings().prompts_dir / version).glob("*.md"))
        raise PromptNotFound(f"no prompt {name!r} in {version}; have: {', '.join(available)}")
    return path.read_text(encoding="utf-8").strip()


def available_prompts(version: str = "v1") -> list[str]:
    directory = settings().prompts_dir / version
    return sorted(p.stem for p in directory.glob("*.md")) if directory.exists() else []


def prompt_fingerprint(version: str = "v1") -> str:
    """Hash of all prompt text, stamped into reports so a run is reproducible."""
    import hashlib

    directory = settings().prompts_dir / version
    digest = hashlib.blake2b(digest_size=8)
    for path in sorted(directory.glob("*.md")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prompts_root() -> Path:
    return settings().prompts_dir
