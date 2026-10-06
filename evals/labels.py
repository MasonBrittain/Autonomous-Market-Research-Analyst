"""Human labelling worksheets.

Calibration needs human labels, and the bottleneck is not the arithmetic -- it is
the tedium of reading fifty claims and their evidence. So the worksheet is built
to make that sitting as cheap as possible: each entry already carries the claim and
the exact quotes it rests on, so a labeller never has to go and find the source.

Deliberately a file to fill in rather than an interactive prompt. Labelling fifty
claims is a half-hour job done in a text editor with the ability to go back and fix
an earlier call -- not something to be marched through one keypress at a time, which
also makes self-consistency impossible to check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from analyst.models import ResearchRun

LABELS_DIR = Path(__file__).parent / "labels"

UNLABELLED = None

INSTRUCTIONS = [
    "Fill in `supported` and `specific` for each entry, then run:",
    "    python -m evals.harness calibrate <run_id>",
    "",
    "supported: do the cited quotes BY THEMSELVES establish the claim?",
    "  true  - the quotes show it, or a short obvious inference from them does",
    "  false - the claim may be true but these quotes do not reach it",
    "",
    "specific: would this sentence be equally true of a random competitor?",
    "  true  - no, it is specific to this company (a figure, date, or named party)",
    "  false - yes, it is filler that would fit anyone",
    "",
    "Leave a field as null to skip that item; skipped items are excluded, not",
    "counted as disagreements. Label at least 50 to clear the calibration gate.",
]


@dataclass
class LabelSet:
    run_id: str
    supported: dict[str, bool]
    specific: dict[str, bool]
    skipped: int = 0

    @property
    def labelled(self) -> int:
        return len(set(self.supported) | set(self.specific))

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "labelled": self.labelled,
            "skipped": self.skipped,
            "supported": len(self.supported),
            "specific": len(self.specific),
        }


def worksheet_path(run_id: str) -> Path:
    return LABELS_DIR / f"{run_id}.json"


def build_worksheet(run: ResearchRun) -> dict[str, object]:
    """One entry per published claim, carrying the evidence it rests on."""
    entries: list[dict[str, object]] = []
    for claim in run.claims:
        if not claim.survived:
            continue
        quotes: list[dict[str, str]] = []
        for fid in claim.fact_ids:
            fact = run.fact_by_id(fid)
            if fact is None:
                continue
            evidence = run.evidence_by_id(fact.evidence_id)
            quotes.append(
                {
                    "fact_id": fact.id,
                    "fact": fact.text,
                    "quote": fact.verbatim_quote,
                    "source": evidence.publisher if evidence else "unknown",
                    "date": fact.happened_at.isoformat() if fact.happened_at else "undated",
                    "url": evidence.url if evidence else "",
                }
            )
        entries.append(
            {
                "claim_id": claim.id,
                "section": claim.section.value
                + (f"/{claim.quadrant.value}" if claim.quadrant else ""),
                "claim": claim.statement,
                "cited_evidence": quotes,
                # Fill these in.
                "supported": UNLABELLED,
                "specific": UNLABELLED,
                "notes": "",
            }
        )
    return {
        "run_id": run.id,
        "query": run.query,
        "entity": run.entity.name if run.entity else None,
        "instructions": INSTRUCTIONS,
        "entries": entries,
    }


def export_worksheet(
    run: ResearchRun, path: Path | None = None, *, overwrite: bool = False
) -> Path:
    """Write a worksheet, refusing by default to clobber existing labels."""
    target = path or worksheet_path(run.id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        existing = load_labels(target)
        if existing.labelled:
            raise FileExistsError(
                f"{target} already holds {existing.labelled} labels. "
                f"Pass overwrite=True only if you mean to discard them."
            )
    target.write_text(json.dumps(build_worksheet(run), indent=2), encoding="utf-8")
    return target


def load_labels(path: Path) -> LabelSet:
    """Read a filled worksheet. Unfilled and malformed entries are skipped."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    supported: dict[str, bool] = {}
    specific: dict[str, bool] = {}
    skipped = 0
    for entry in payload.get("entries", []):
        claim_id = entry.get("claim_id")
        if not claim_id:
            continue
        got_one = False
        for field_name, target in (("supported", supported), ("specific", specific)):
            value = entry.get(field_name)
            if isinstance(value, bool):
                target[claim_id] = value
                got_one = True
        if not got_one:
            skipped += 1
    return LabelSet(
        run_id=str(payload.get("run_id", "")),
        supported=supported,
        specific=specific,
        skipped=skipped,
    )


def claim_context(run: ResearchRun) -> dict[str, str]:
    """Short claim text per id, so calibration disagreements are readable."""
    return {c.id: c.statement[:160] for c in run.claims}


def label_progress(run_id: str) -> str:
    path = worksheet_path(run_id)
    if not path.exists():
        return f"no worksheet for {run_id}; run: python -m evals.harness label {run_id}"
    labels = load_labels(path)
    return (
        f"{labels.labelled} labelled, {labels.skipped} still blank "
        f"(supported={len(labels.supported)}, specific={len(labels.specific)})"
    )
