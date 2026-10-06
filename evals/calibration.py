"""Judge calibration: how much should we trust the judge's numbers?

An LLM judge that has never been checked against human labels produces a number
with a decimal point and no meaning. This module is the check.

Three things are computed, and all three matter:

* **Raw agreement** -- the share of items the judge and the human labelled the
  same way. Looks impressive and is almost always misleading on its own, because
  a judge that answers "supported" every single time scores 0.9 agreement against
  a set that is 90% supported.
* **Cohen's kappa** -- agreement corrected for what chance alone would produce.
  This is the number that actually says whether the judge is doing work.
* **A Wilson interval** on every reported proportion. Calibration sets are small
  (50 items is a realistic ceiling for hand labelling), and a point estimate from
  n=50 with no interval invites over-reading.

`gate()` then refuses to call a judge trustworthy until the sample is big enough
and kappa clears a floor. The point of the gate is to make "we did not verify
this" the default state rather than something you have to remember to say.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

# Sample-size and kappa floors for treating judged metrics as quotable.
MIN_CALIBRATION_SAMPLE = 50
MIN_KAPPA = 0.6


class Trust(str, Enum):
    """How much weight a judged metric can carry."""

    UNCALIBRATED = "uncalibrated"  # no human labels at all
    INSUFFICIENT = "insufficient"  # labelled, but too few items
    UNRELIABLE = "unreliable"  # enough items, kappa too low
    TRUSTED = "trusted"  # clears both floors


@dataclass(frozen=True)
class Interval:
    """A Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sanely at 0 and 1 -- both of which happen constantly at n=50.
    """

    point: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        if self.n == 0:
            return "n/a (n=0)"
        return f"{self.point:.3f} [{self.low:.3f}-{self.high:.3f}] n={self.n}"

    @property
    def width(self) -> float:
        return round(self.high - self.low, 4)


def wilson(successes: int, n: int, z: float = 1.96) -> Interval:
    """95% Wilson score interval by default."""
    if n <= 0:
        return Interval(point=0.0, low=0.0, high=0.0, n=0)
    successes = max(0, min(successes, n))
    p = successes / n
    denom = 1 + (z**2) / n
    centre = p + (z**2) / (2 * n)
    spread = z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
    return Interval(
        point=round(p, 4),
        low=round(max(0.0, (centre - spread) / denom), 4),
        high=round(min(1.0, (centre + spread) / denom), 4),
        n=n,
    )


@dataclass
class ConfusionMatrix:
    """Judge predictions against human labels for one binary question."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    @property
    def n(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @property
    def agreement(self) -> Interval:
        return wilson(self.true_positive + self.true_negative, self.n)

    @property
    def precision(self) -> Interval:
        """Of the items the judge called positive, how many the human did too."""
        return wilson(self.true_positive, self.true_positive + self.false_positive)

    @property
    def recall(self) -> Interval:
        """Of the items the human called positive, how many the judge caught."""
        return wilson(self.true_positive, self.true_positive + self.false_negative)

    @property
    def kappa(self) -> float:
        """Cohen's kappa. 0 means chance-level, 1 means perfect.

        Convention at the degenerate end: when every cell but one is empty,
        chance agreement is 1.0 and kappa is undefined. We report 1.0 if the two
        raters agreed completely and 0.0 otherwise, which is the reading that
        cannot flatter a judge.
        """
        n = self.n
        if n == 0:
            return 0.0
        observed = (self.true_positive + self.true_negative) / n
        judge_pos = (self.true_positive + self.false_positive) / n
        human_pos = (self.true_positive + self.false_negative) / n
        expected = (judge_pos * human_pos) + ((1 - judge_pos) * (1 - human_pos))
        if expected >= 1.0:
            return 1.0 if observed >= 1.0 else 0.0
        return round((observed - expected) / (1 - expected), 4)

    @property
    def kappa_label(self) -> str:
        """Landis & Koch bands, so the number reads without a lookup."""
        k = self.kappa
        if k < 0.0:
            return "worse than chance"
        if k <= 0.0:
            # The case kappa exists to expose: a judge that answers the same way
            # every time scores high raw agreement against a skewed set while
            # adding nothing.
            return "chance level"
        if k < 0.21:
            return "slight"
        if k < 0.41:
            return "fair"
        if k < 0.61:
            return "moderate"
        if k < 0.81:
            return "substantial"
        return "almost perfect"

    def add(self, judge_says: bool, human_says: bool) -> None:
        if judge_says and human_says:
            self.true_positive += 1
        elif judge_says and not human_says:
            self.false_positive += 1
        elif not judge_says and human_says:
            self.false_negative += 1
        else:
            self.true_negative += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "agreement": str(self.agreement),
            "precision": str(self.precision),
            "recall": str(self.recall),
            "kappa": self.kappa,
            "kappa_label": self.kappa_label,
        }


@dataclass
class Calibration:
    """Calibration result for one judged question."""

    question: str
    matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    disagreements: list[dict[str, object]] = field(default_factory=list)

    @property
    def trust(self) -> Trust:
        if self.matrix.n == 0:
            return Trust.UNCALIBRATED
        if self.matrix.n < MIN_CALIBRATION_SAMPLE:
            return Trust.INSUFFICIENT
        if self.matrix.kappa < MIN_KAPPA:
            return Trust.UNRELIABLE
        return Trust.TRUSTED

    @property
    def quotable(self) -> bool:
        return self.trust is Trust.TRUSTED

    def caveat(self) -> str:
        """The sentence that must accompany this metric wherever it is reported."""
        m = self.matrix
        if self.trust is Trust.UNCALIBRATED:
            return (
                f"{self.question}: NOT CALIBRATED -- no human labels. "
                f"Judge output is unvalidated and must not be quoted as a result."
            )
        if self.trust is Trust.INSUFFICIENT:
            return (
                f"{self.question}: calibrated on only {m.n} items "
                f"(need {MIN_CALIBRATION_SAMPLE}). Agreement {m.agreement}, "
                f"kappa {m.kappa} ({m.kappa_label}). Treat as indicative only."
            )
        if self.trust is Trust.UNRELIABLE:
            return (
                f"{self.question}: kappa {m.kappa} ({m.kappa_label}) is below the "
                f"{MIN_KAPPA} floor on n={m.n}. The judge does not agree with human "
                f"labels well enough to stand in for them."
            )
        return (
            f"{self.question}: judge agrees with human labels {m.agreement}, "
            f"kappa {m.kappa} ({m.kappa_label}) on n={m.n}."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "trust": self.trust.value,
            "quotable": self.quotable,
            "caveat": self.caveat(),
            **self.matrix.as_dict(),
            "disagreements": self.disagreements,
        }


def calibrate(
    question: str,
    judge_verdicts: dict[str, bool],
    human_labels: dict[str, bool],
    context: dict[str, str] | None = None,
) -> Calibration:
    """Compare judge verdicts to human labels over their shared keys.

    Only items present in both are scored. Keeping the disagreements is the
    practically useful part -- they are what you read to work out whether the
    judge prompt is wrong or your own labelling was inconsistent.
    """
    result = Calibration(question=question)
    for key in sorted(set(judge_verdicts) & set(human_labels)):
        judge_says = bool(judge_verdicts[key])
        human_says = bool(human_labels[key])
        result.matrix.add(judge_says, human_says)
        if judge_says != human_says:
            result.disagreements.append(
                {
                    "id": key,
                    "judge": judge_says,
                    "human": human_says,
                    "context": (context or {}).get(key, ""),
                }
            )
    return result


def gate(calibrations: list[Calibration]) -> tuple[bool, list[str]]:
    """Whether judged metrics may be reported as results, and why not if not."""
    blockers = [c.caveat() for c in calibrations if not c.quotable]
    return (not blockers, blockers)
