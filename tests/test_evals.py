"""Eval-layer tests: calibration arithmetic, the specificity detector, the judge,
worksheet round-trips, and the gate that keeps unverified numbers unquoted.

All offline. The judge runs against `JudgeStub`, so the grading, calibration and
reporting paths are exercised with no API key.
"""

from __future__ import annotations

import json

import pytest

from evals import labels as labels_mod
from evals import specificity
from evals.calibration import (
    MIN_CALIBRATION_SAMPLE,
    MIN_KAPPA,
    Calibration,
    ConfusionMatrix,
    Trust,
    calibrate,
    gate,
    wilson,
)
from evals.harness import GoldenTarget, score_structural
from evals.judge import Judge
from evals.semantic import score_semantic
from evals.stub import JudgeStub

# --------------------------------------------------------------------------- #
# Wilson intervals
# --------------------------------------------------------------------------- #


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    """The reason for Wilson over the normal approximation: 10/10 and 0/10 are
    both common at calibration-set sizes and both break the naive formula."""
    perfect = wilson(10, 10)
    assert perfect.point == 1.0
    assert perfect.high == 1.0
    assert perfect.low < 1.0, "a perfect run of 10 is not proof of perfection"

    none = wilson(0, 10)
    assert none.point == 0.0
    assert none.low == 0.0
    assert none.high > 0.0


def test_wilson_interval_narrows_as_n_grows():
    small = wilson(9, 10)
    large = wilson(90, 100)
    assert small.point == large.point
    assert large.width < small.width, "more evidence must mean a tighter interval"


def test_wilson_handles_zero_sample():
    empty = wilson(0, 0)
    assert empty.n == 0
    assert "n/a" in str(empty)


def test_wilson_clamps_successes_above_n():
    assert wilson(12, 10).point == 1.0


# --------------------------------------------------------------------------- #
# Cohen's kappa
# --------------------------------------------------------------------------- #


def test_kappa_exposes_a_judge_that_always_agrees():
    """The headline reason raw agreement is not reported alone.

    A judge answering "yes" to everything scores 0.9 agreement against a set that
    is 90% positive, while contributing nothing. Kappa must call that chance-level.
    """
    rubber_stamp = ConfusionMatrix(true_positive=90, false_positive=10)
    assert rubber_stamp.agreement.point == 0.9, "raw agreement looks good"
    assert rubber_stamp.kappa == 0.0, "kappa must see through it"
    assert rubber_stamp.kappa_label == "chance level"


def test_kappa_rewards_real_agreement():
    good = ConfusionMatrix(true_positive=45, true_negative=45, false_positive=5, false_negative=5)
    assert good.kappa == pytest.approx(0.8, abs=0.01)
    assert good.kappa_label == "substantial"


def test_kappa_perfect_and_inverted():
    assert ConfusionMatrix(true_positive=50, true_negative=50).kappa == 1.0
    inverted = ConfusionMatrix(false_positive=50, false_negative=50)
    assert inverted.kappa == -1.0
    assert inverted.kappa_label == "worse than chance"


def test_kappa_degenerate_single_cell_does_not_flatter():
    """Everything in one cell makes chance agreement 1.0 and kappa undefined.
    Full agreement reads 1.0; anything else reads 0.0, never something in between."""
    all_agree = ConfusionMatrix(true_positive=20)
    assert all_agree.kappa == 1.0
    assert ConfusionMatrix(false_positive=20).kappa == 0.0


def test_kappa_empty_matrix_is_zero():
    assert ConfusionMatrix().kappa == 0.0
    assert ConfusionMatrix().n == 0


def test_precision_and_recall_read_off_the_right_axes():
    m = ConfusionMatrix(true_positive=8, false_positive=2, false_negative=4, true_negative=6)
    assert m.precision.point == pytest.approx(0.8)
    assert m.recall.point == pytest.approx(0.667, abs=0.001)


# --------------------------------------------------------------------------- #
# The calibration gate
# --------------------------------------------------------------------------- #


def _matrix_with(n: int, kappa_target: str) -> ConfusionMatrix:
    """Build a matrix of size n with either strong or chance-level kappa.

    The low case has to be *skewed and imperfect* -- a rubber-stamp judge against
    a mostly-positive set. An all-in-one-cell matrix is full agreement, which
    scores 1.0, not 0.
    """
    if kappa_target == "high":
        half = n // 2
        return ConfusionMatrix(true_positive=half, true_negative=n - half)
    positives = int(n * 0.9)
    return ConfusionMatrix(true_positive=positives, false_positive=n - positives)


def test_no_labels_means_uncalibrated_and_unquotable():
    cal = Calibration(question="citation precision")
    assert cal.trust is Trust.UNCALIBRATED
    assert not cal.quotable
    assert "NOT CALIBRATED" in cal.caveat()


def test_small_sample_is_insufficient_however_good_the_agreement():
    cal = Calibration(question="q", matrix=_matrix_with(20, "high"))
    assert cal.matrix.kappa == 1.0
    assert cal.trust is Trust.INSUFFICIENT, "perfect agreement on 20 items is still 20 items"
    assert not cal.quotable
    assert str(MIN_CALIBRATION_SAMPLE) in cal.caveat()


def test_large_sample_with_low_kappa_is_unreliable():
    cal = Calibration(question="q", matrix=_matrix_with(100, "low"))
    assert cal.matrix.n >= MIN_CALIBRATION_SAMPLE
    assert cal.matrix.kappa < MIN_KAPPA
    assert cal.trust is Trust.UNRELIABLE
    assert not cal.quotable


def test_trusted_only_when_both_floors_clear():
    cal = Calibration(question="q", matrix=_matrix_with(MIN_CALIBRATION_SAMPLE, "high"))
    assert cal.trust is Trust.TRUSTED
    assert cal.quotable
    assert "agrees with human labels" in cal.caveat()


def test_gate_blocks_on_the_weakest_question():
    strong = Calibration(question="a", matrix=_matrix_with(MIN_CALIBRATION_SAMPLE, "high"))
    weak = Calibration(question="b")
    ok, blockers = gate([strong, weak])
    assert not ok
    assert len(blockers) == 1 and "b" in blockers[0]

    ok_all, blockers_all = gate([strong])
    assert ok_all and not blockers_all


def test_calibrate_scores_only_shared_keys_and_keeps_disagreements():
    judge = {"a": True, "b": True, "c": False, "orphan": True}
    human = {"a": True, "b": False, "c": False, "unlabelled": True}
    cal = calibrate("q", judge, human, context={"b": "the disputed claim"})

    assert cal.matrix.n == 3, "items missing from either side must be excluded"
    assert len(cal.disagreements) == 1
    disagreement = cal.disagreements[0]
    assert disagreement["id"] == "b"
    assert disagreement["judge"] is True and disagreement["human"] is False
    assert disagreement["context"] == "the disputed claim"


# --------------------------------------------------------------------------- #
# Specificity detector
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "statement",
    [
        "Gross margin of 46.2% is 900bps above the peer median.",
        "Revenue rose to $4.3 billion in fiscal 2026.",
        "Net leverage fell to 2.1x trailing earnings.",
        "The DOJ opened an inquiry in March.",
    ],
)
def test_detector_accepts_claims_carrying_a_figure_or_date(statement):
    assert specificity.assess(statement).specific, statement


@pytest.mark.parametrize(
    "statement",
    [
        "The company has a strong brand and experienced management.",
        "Exposed to macroeconomic conditions and increased competition.",
        "Well positioned to capture growth opportunities.",
        "A robust balance sheet supports strategic initiatives.",
    ],
)
def test_detector_rejects_filler(statement):
    verdict = specificity.assess(statement)
    assert not verdict.specific, statement
    assert verdict.filler_hits


def test_detector_accepts_filler_once_a_figure_is_attached():
    """Filler phrasing is not itself disqualifying -- vacuity is."""
    assert specificity.assess(
        "Strong brand recognition supported a 46% gross margin, 900bps above peers."
    ).specific


def test_detector_counts_a_named_third_party():
    verdict = specificity.assess(
        "Loses share to Caterpillar in the compact equipment segment.", subject="Acme Corporation"
    )
    assert verdict.specific
    assert "Caterpillar" in verdict.named_entities


def test_detector_does_not_count_the_subject_as_a_third_party():
    verdict = specificity.assess(
        "Acme continues to serve its customers well.", subject="Acme Corporation"
    )
    assert "Acme" not in verdict.named_entities
    assert not verdict.specific


def test_detector_ignores_sentence_initial_capitals():
    assert specificity.named_entities("Margins improved across the business.") == []


def test_ordinal_alone_is_not_a_figure():
    """'largest' with no number is a boast, not a measurement."""
    verdict = specificity.assess("The largest player in its category.")
    assert "ordinal_rank" in verdict.markers
    assert not verdict.specific


def test_rate_counts_over_a_list():
    specific, total = specificity.rate(
        ["Revenue rose 12% to $4.3B.", "Strong brand and market leader."]
    )
    assert (specific, total) == (1, 2)


# --------------------------------------------------------------------------- #
# Judge (against the offline stand-in)
# --------------------------------------------------------------------------- #


def test_judge_grades_every_published_claim(sample_run):
    result = Judge(JudgeStub()).grade_citations(sample_run)
    published = [c for c in sample_run.claims if c.survived]

    assert result.total == len(published)
    assert set(result.grades) == {c.id for c in published}
    assert all(g.reasoning for g in result.grades.values())
    assert 0 <= result.established <= result.total


def test_judge_never_grades_a_rejected_claim(sample_run):
    """Rejected claims are not in the brief, so grading them would be measuring
    work the pipeline already discarded."""
    rejected = [c for c in sample_run.claims if not c.survived]
    assert rejected, "fixture should contain a rejected claim"
    result = Judge(JudgeStub()).grade_citations(sample_run)
    for claim in rejected:
        assert claim.id not in result.grades


def test_judge_respects_the_limit(sample_run):
    result = Judge(JudgeStub()).grade_citations(sample_run, limit=2)
    assert result.total == 2


def test_citation_failures_carry_the_unsupported_part(sample_run):
    result = Judge(JudgeStub()).grade_citations(sample_run)
    for grade in result.failures().values():
        assert not grade.established
        assert grade.unsupported_part


def test_coverage_grades_every_reference_point(sample_run):
    points = [
        "Revenue grew 12% with margin expansion.",
        "Manufacturing is concentrated in one region.",
        "A federal inquiry into supplier certification is open.",
        "Guidance was raised for the full year.",
        "Customer concentration is unquantified.",
    ]
    result = Judge(JudgeStub()).grade_coverage(sample_run, points)

    assert result.total == len(points)
    assert [p.point_index for p in result.graded] == list(range(len(points)))
    assert 0 <= result.covered <= result.total
    for point, why in result.misses():
        assert point in points
        assert why


def test_coverage_with_no_expected_points_is_empty_not_perfect(sample_run):
    """An unlabelled target must not score 100% by vacuous truth."""
    result = Judge(JudgeStub()).grade_coverage(sample_run, [])
    assert result.total == 0
    assert result.covered == 0
    assert wilson(result.covered, result.total).n == 0


def test_coverage_marks_everything_missed_when_nothing_was_published(sample_run):
    for claim in sample_run.claims:
        assert claim.verdict is not None
        claim.verdict.verdict = claim.verdict.verdict.__class__("reject")
    result = Judge(JudgeStub()).grade_coverage(sample_run, ["a point", "another point"])
    assert result.total == 2
    assert result.covered == 0


def test_coverage_rejects_a_match_against_a_nonexistent_claim(sample_run):
    """A judge citing a claim id that is not in the brief cannot be trusted to
    have matched anything, so the point counts as missed."""
    from evals.judge import CoverageGrade, CoveragePoint, SpecificityGrade  # noqa: F401

    class FabricatesIds(JudgeStub):
        def _make_coverage_grade(self, node, system, user):
            return CoverageGrade(
                points=[
                    CoveragePoint(
                        point_index=0,
                        covered=True,
                        matching_claim_id="c_ffffffffffff",
                        reasoning="invented",
                    )
                ]
            )

    result = Judge(FabricatesIds()).grade_coverage(sample_run, ["a point"])
    assert result.covered == 0
    assert "nonexistent" in result.graded[0].reasoning


def test_specificity_runs_detector_and_judge_over_the_same_claims(sample_run):
    result = Judge(JudgeStub()).grade_specificity(sample_run)
    published = {c.id for c in sample_run.claims if c.survived}

    assert set(result.detector) == published
    assert set(result.judge) <= published
    assert 0.0 <= result.detector_rate <= 1.0
    assert 0.0 <= result.judge_rate <= 1.0
    for cid in result.disagreements:
        assert result.detector[cid] != result.judge[cid]


def test_judge_cost_is_accounted(sample_run):
    from evals.judge import total_cost

    citations = Judge(JudgeStub()).grade_citations(sample_run)
    assert citations.usage
    assert total_cost(citations) > 0, "grading is real spend and must be reported"


# --------------------------------------------------------------------------- #
# Worksheets
# --------------------------------------------------------------------------- #


def test_worksheet_carries_the_evidence_so_labelling_needs_no_lookup(sample_run, tmp_path):
    path = labels_mod.export_worksheet(sample_run, tmp_path / "ws.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    published = [c for c in sample_run.claims if c.survived]
    assert len(payload["entries"]) == len(published)
    assert payload["instructions"]
    for entry in payload["entries"]:
        assert entry["supported"] is None, "fields start unlabelled"
        assert entry["specific"] is None
        assert entry["cited_evidence"], "an entry with no quotes cannot be labelled"
        for citation in entry["cited_evidence"]:
            assert citation["quote"]
            assert citation["source"]


def test_worksheet_excludes_rejected_claims(sample_run, tmp_path):
    path = labels_mod.export_worksheet(sample_run, tmp_path / "ws.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = {e["claim_id"] for e in payload["entries"]}
    for claim in sample_run.claims:
        if not claim.survived:
            assert claim.id not in ids


def test_load_labels_reads_filled_fields_and_skips_blanks(sample_run, tmp_path):
    path = labels_mod.export_worksheet(sample_run, tmp_path / "ws.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["supported"] = True
    payload["entries"][0]["specific"] = False
    payload["entries"][1]["supported"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    labels = labels_mod.load_labels(path)
    assert len(labels.supported) == 2
    assert len(labels.specific) == 1
    assert labels.skipped == len(payload["entries"]) - 2
    assert labels.labelled == 2


def test_load_labels_ignores_non_boolean_values(sample_run, tmp_path):
    path = labels_mod.export_worksheet(sample_run, tmp_path / "ws.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["supported"] = "yes"  # a string, not a label
    payload["entries"][1]["supported"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    labels = labels_mod.load_labels(path)
    assert labels.supported == {}, "only real booleans count as labels"


def test_export_refuses_to_clobber_existing_labels(sample_run, tmp_path):
    path = labels_mod.export_worksheet(sample_run, tmp_path / "ws.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"][0]["supported"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FileExistsError, match="already holds"):
        labels_mod.export_worksheet(sample_run, path)

    # Explicit opt-in still works.
    labels_mod.export_worksheet(sample_run, path, overwrite=True)
    assert labels_mod.load_labels(path).labelled == 0


def test_export_overwrites_an_unlabelled_worksheet_without_complaint(sample_run, tmp_path):
    path = labels_mod.export_worksheet(sample_run, tmp_path / "ws.json")
    labels_mod.export_worksheet(sample_run, path)  # no labels yet, so no guard


# --------------------------------------------------------------------------- #
# Semantic scoring end to end
# --------------------------------------------------------------------------- #


def test_semantic_score_is_unquotable_without_labels(sample_run):
    score = score_semantic(sample_run, JudgeStub())

    assert score.citation_precision.n > 0, "claims were graded"
    assert score.trust is Trust.UNCALIBRATED
    assert not score.quotable
    text = "\n".join(score.summary_lines())
    assert "NOT QUOTABLE" in text
    assert "no human labels" in text


def test_semantic_score_reports_numbers_with_intervals(sample_run):
    score = score_semantic(sample_run, JudgeStub())
    text = "\n".join(score.summary_lines())
    # Every proportion is printed with its interval and n, never bare.
    assert "[" in text and "n=" in text
    assert score.judge_cost_usd > 0


def test_semantic_score_serializes(sample_run):
    payload = score_semantic(sample_run, JudgeStub()).as_dict()
    assert payload["trust"] == "uncalibrated"
    assert payload["quotable"] is False
    assert "citation_precision" in payload
    assert set(payload["citation_precision"]) == {"point", "low", "high", "n"}


def test_golden_targets_load_and_declare_their_kind():
    targets = GoldenTarget.load_all()
    assert targets, "golden set is empty"
    assert any(t.should_be_ambiguous for t in targets), "no adversarial target"
    assert any(t.should_be_industry for t in targets), "no industry target"
    for target in targets:
        assert target.query
        assert target.notes, "a golden target without notes is unmaintainable"


def test_structural_and_semantic_cover_different_questions(sample_run):
    """Guard against the two layers silently measuring the same thing."""
    structural = score_structural(sample_run)
    semantic = score_semantic(sample_run, JudgeStub())

    # Structural asks "is the citation present and real"; it is satisfied here.
    assert structural.citation_validity == 1.0
    assert structural.dangling_citations == 0
    # Semantic asks "does the cited evidence actually establish the claim" --
    # a strictly harder question, so it may legitimately score lower.
    assert semantic.citation_precision.n == structural.claims_published
