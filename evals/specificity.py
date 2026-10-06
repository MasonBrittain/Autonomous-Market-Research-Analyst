"""Automated half of the specificity metric.

The question "would this sentence be equally true of a random competitor?" is a
judgement call, but a large part of it is mechanical: a claim carrying a figure, a
date, or a named third party is specific whatever else is wrong with it, and a
claim built out of analyst-deck filler is not.

Running the mechanical part first is worth doing for its own sake -- it is free,
deterministic, and needs no calibration -- and it also gives the judge something
to be checked against. Where the detector and the judge disagree, one of them is
wrong, and that is a useful place to look.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Quantitative markers. Any one of these makes a claim concrete.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "percentage": re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent|pct)\b", re.I),
    "basis_points": re.compile(r"\b\d+\s?(?:bps|basis points)\b", re.I),
    "currency": re.compile(
        r"[$€£¥]\s?\d|\b\d+(?:\.\d+)?\s?(?:billion|million|trillion|bn|mm?)\b", re.I
    ),
    "multiple": re.compile(r"\b\d+(?:\.\d+)?\s?(?:x|times)\b", re.I),
    "ordinal_rank": re.compile(
        r"\b(?:first|second|third|fourth|largest|smallest|top|bottom)\b", re.I
    ),
    "date": re.compile(
        r"\b(?:Q[1-4]\s?(?:20\d\d|FY\d\d)|20\d\d|fiscal\s+20\d\d|"
        r"January|February|March|April|May|June|July|August|September|October|November|December)\b"
    ),
    "bare_number": re.compile(r"\b\d{2,}\b"),
}

# Phrases that fit any company in the index. Presence does not condemn a claim on
# its own -- "strong brand recognition supported 46% gross margin" is fine -- but a
# claim made only of these is filler.
_FILLER = (
    "strong brand",
    "brand recognition",
    "experienced management",
    "experienced leadership",
    "strong management team",
    "market leader",
    "industry leader",
    "well positioned",
    "well-positioned",
    "competitive landscape",
    "operational efficiency",
    "operational excellence",
    "economies of scale",
    "synergies",
    "industry headwinds",
    "macroeconomic conditions",
    "macroeconomic uncertainty",
    "economic uncertainty",
    "changing consumer preferences",
    "regulatory environment",
    "technological change",
    "increased competition",
    "global presence",
    "diversified portfolio",
    "robust balance sheet",
    "customer loyalty",
    "innovative products",
    "digital transformation",
    "growth opportunities",
    "strategic initiatives",
)

_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)([A-Z])")
_CAPITALISED = re.compile(r"\b[A-Z][A-Za-z&.\-]{2,}\b")

# Words that are capitalised for reasons other than being a named third party.
_NOT_AN_ENTITY = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "Its",
    "Their",
    "A",
    "An",
    "While",
    "However",
    "Although",
    "Despite",
    "Management",
    "Company",
    "Revenue",
    "Gross",
    "Operating",
    "Net",
    "Free",
    "Cash",
    "Analysts",
    "Chief",
    "Executive",
    "Officer",
    "Item",
    "Risk",
    "Factors",
    "SEC",
    "GAAP",
    "US",
    "U.S.",
    "North",
    "America",
    "Europe",
    "Asia",
    "China",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "FY",
}


@dataclass
class SpecificityVerdict:
    specific: bool
    markers: list[str]
    filler_hits: list[str]
    named_entities: list[str]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "specific": self.specific,
            "markers": self.markers,
            "filler_hits": self.filler_hits,
            "named_entities": self.named_entities,
            "reason": self.reason,
        }


def named_entities(statement: str, subject: str = "") -> list[str]:
    """Capitalised tokens that are plausibly a third party, not the subject.

    Deliberately crude -- the goal is "does this claim name anyone concrete",
    not full NER. A real entity recogniser would be more accurate and would also
    be a dependency and a model call for a signal this coarse.
    """
    subject_tokens = {t.lower() for t in re.findall(r"[A-Za-z]+", subject)} if subject else set()
    starts = {m.start(1) for m in _SENTENCE_START.finditer(statement)}
    found: list[str] = []
    for match in _CAPITALISED.finditer(statement):
        token = match.group(0)
        if match.start() in starts:
            continue  # capitalised only because it opens a sentence
        if token in _NOT_AN_ENTITY:
            continue
        if token.lower().strip(".") in subject_tokens:
            continue  # the subject itself is not a third party
        if token not in found:
            found.append(token)
    return found


def assess(statement: str, subject: str = "") -> SpecificityVerdict:
    """Mechanical specificity verdict for one claim."""
    text = statement.strip()
    markers = [name for name, pattern in _PATTERNS.items() if pattern.search(text)]
    lowered = text.lower()
    filler_hits = [phrase for phrase in _FILLER if phrase in lowered]
    entities = named_entities(text, subject)

    # "ordinal_rank" alone is weak -- "largest" with no figure is still a boast --
    # so it does not count as a quantitative marker by itself.
    hard_markers = [m for m in markers if m != "ordinal_rank"]

    if hard_markers:
        return SpecificityVerdict(
            specific=True,
            markers=markers,
            filler_hits=filler_hits,
            named_entities=entities,
            reason=f"carries {', '.join(hard_markers)}",
        )
    if entities and not filler_hits:
        return SpecificityVerdict(
            specific=True,
            markers=markers,
            filler_hits=filler_hits,
            named_entities=entities,
            reason=f"names a third party ({', '.join(entities[:3])}) with no filler",
        )
    if filler_hits:
        return SpecificityVerdict(
            specific=False,
            markers=markers,
            filler_hits=filler_hits,
            named_entities=entities,
            reason=f"filler with no figure: {', '.join(filler_hits[:3])}",
        )
    return SpecificityVerdict(
        specific=False,
        markers=markers,
        filler_hits=filler_hits,
        named_entities=entities,
        reason="no figure, date, or named third party",
    )


def rate(statements: list[str], subject: str = "") -> tuple[int, int]:
    """(specific_count, total) over a list of claims."""
    verdicts = [assess(s, subject) for s in statements]
    return sum(1 for v in verdicts if v.specific), len(verdicts)
