"""Exact and near-duplicate detection over fetched documents.

News syndication means one Reuters story arrives through eight outlets with
different headlines and boilerplate. Left alone, those eight copies make a claim
look eight times better evidenced than it is -- so deduplication is a
correctness concern here, not a tidiness one.

SimHash is implemented directly rather than pulled from a dependency: it is
~40 lines, it keeps the install light, and the threshold needs tuning against
our own corpus anyway.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

from ..models import Evidence

_WORD = re.compile(r"[a-z0-9']+")
HASH_BITS = 64


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def shingles(text: str, n: int = 3) -> list[str]:
    """Word n-grams. Shingles beat bare words: they capture local word order,
    so two articles about the same event score far closer than two articles that
    merely share a vocabulary."""
    words = _tokens(text)
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def simhash(text: str, n: int = 3) -> int:
    """64-bit locality-sensitive fingerprint: similar text -> similar fingerprint."""
    grams = shingles(text, n)
    if not grams:
        return 0
    weights = [0] * HASH_BITS
    for gram in grams:
        h = _hash64(gram)
        for bit in range(HASH_BITS):
            weights[bit] += 1 if (h >> bit) & 1 else -1
    fingerprint = 0
    for bit in range(HASH_BITS):
        if weights[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def similarity(a: str, b: str) -> float:
    """Convenience 0..1 score, mostly for tests and threshold tuning."""
    return 1.0 - (hamming(simhash(a), simhash(b)) / HASH_BITS)


def _representative(group: Sequence[Evidence]) -> Evidence:
    """Pick the canonical member of a duplicate cluster.

    Earliest publication wins (it is likeliest to be the original reporting),
    then source authority, then length as a proxy for completeness.
    """

    def sort_key(e: Evidence) -> tuple[float, float, int]:
        ts = e.published_at.timestamp() if e.published_at else float("inf")
        return (ts, -e.authority, -len(e.clean_text))

    return sorted(group, key=sort_key)[0]


def deduplicate(
    evidence: Iterable[Evidence], max_distance: int = 8, shingle_n: int = 3
) -> tuple[list[Evidence], int]:
    """Mark duplicates in place and return (evidence, clusters_found).

    Each duplicate keeps a `duplicate_of` pointer rather than being dropped, so
    the report can still say "corroborated by 6 outlets" without double-counting
    the underlying fact.
    """
    items = [e for e in evidence]
    by_exact: dict[str, Evidence] = {}
    survivors: list[Evidence] = []

    # Pass 1: exact body hash.
    for e in items:
        if not e.hash or not e.clean_text.strip():
            survivors.append(e)
            continue
        prior = by_exact.get(e.hash)
        if prior is not None:
            e.duplicate_of = prior.id
            e.cluster_id = prior.cluster_id or prior.id
        else:
            by_exact[e.hash] = e
            survivors.append(e)

    # Pass 2: near-duplicate clustering over whatever survived pass 1.
    fingerprints: list[tuple[Evidence, int]] = [
        (e, simhash(e.clean_text, shingle_n)) for e in survivors if e.clean_text.strip()
    ]
    clusters: list[list[tuple[Evidence, int]]] = []
    for item in fingerprints:
        placed = False
        for cluster in clusters:
            if hamming(item[1], cluster[0][1]) <= max_distance:
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    multi = 0
    for cluster in clusters:
        members = [e for e, _ in cluster]
        if len(members) == 1:
            continue
        multi += 1
        rep = _representative(members)
        rep.cluster_id = rep.id
        for member in members:
            if member.id == rep.id:
                continue
            member.duplicate_of = rep.id
            member.cluster_id = rep.id

    return items, multi


def cluster_sizes(evidence: Iterable[Evidence]) -> dict[str, int]:
    """How many outlets carried each surviving story. Feeds corroboration counts."""
    sizes: dict[str, int] = {}
    for e in evidence:
        root = e.duplicate_of or e.id
        sizes[root] = sizes.get(root, 0) + 1
    return sizes
