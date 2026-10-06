# Eval harness

Two kinds of metric, kept deliberately separate, because a scoreboard that mixes an
exact number with a model's opinion — without saying which is which — is worse than
no scoreboard.

```bash
python -m evals.harness report              # structural scoreboard (no API key)
python -m evals.harness judge <run_id>      # judged metrics (--live for the real model)
python -m evals.harness label <run_id>      # export a human-labelling worksheet
python -m evals.harness calibrate <run_id>  # judge vs. human labels
python -m evals.harness semantic-status     # what is currently blocking judged metrics
python -m evals.harness golden              # list the golden set
```

## Structural metrics

Computed from a stored run with no model involved. Cheap, exact, and run in CI.

| Metric | Target | Meaning |
| --- | --- | --- |
| `citation_validity` | 1.00 | Every published claim cites at least one fact. Below 1.0 is a bug. |
| `quote_integrity` | 1.00 | Every fact's quote is a literal substring of its source. Below 1.0 is a bug. |
| `dangling_citations` | 0 | No claim cites a fact outside the pack. |
| `coverage_score` | tune | Share of Scout's seven dimensions evidenced. |
| `dedup_rate` | tune | Share of gathered documents found to be duplicates. |
| `rejection_rate` | > 0 | Share of claims the Adversary rejected. Zero means the pass is not working. |
| `cost_usd` | tune | Real spend per brief. |
| `cache_read_tokens` | > 0 | Proof the cached fact-pack prefix is being reused. |

Runs that did not complete report `n/a` and are excluded from the integrity gate — a
run that correctly stopped on an ambiguous target has nothing to check, and scoring it
zero would show a regression where the pipeline behaved as designed.

## Judged metrics

| Metric | Question |
| --- | --- |
| `citation_precision` | Do a claim's cited facts actually *establish* it? |
| `golden_coverage` | What share of a target's hand-written `expected_points` did the brief reach? |
| `specificity` | Share of claims that are not generic filler. |

Three things about these are worth understanding before reading any of their numbers.

**They are gated on calibration.** An LLM judge that has never been checked against
human labels produces a number with a decimal point and no meaning. Until a run has
hand labels, its judged metrics report `trust: uncalibrated` and `quotable: false`, and
the blocking reason prints alongside the figure. This is enforced in code rather than
left to anyone's memory.

**Raw agreement is never reported alone.** A judge that answers "supported" every time
scores 0.90 agreement against a set that is 90% supported, while doing no work at all.
Cohen's kappa catches exactly that — the same matrix scores kappa 0.0, labelled *chance
level*. The gate requires kappa ≥ 0.6 on ≥ 50 labelled items.

**Every proportion carries a Wilson interval.** Calibration sets are small; 50 items is
a realistic ceiling for hand labelling. `0.933 [0.702-0.988] n=15` is honest in a way
that `93.3%` is not.

### One caveat stated plainly

Citation precision measures **residual** error, not error in the raw analysis. The
production Adversary already filters claims on very nearly the same question, so the
judge is grading what survived that filter. Worse, judge and Adversary share a model,
so their mistakes correlate. The judge prompt is framed differently on purpose — an
outside auditor grading a finished brief, rather than a reviewer improving a
teammate's work — but framing only decorrelates them a little. Calibration against
human labels is the real defence.

### Specificity is measured twice

A mechanical detector (figures, dates, basis points, multiples, named third parties,
and a filler-phrase list) runs alongside the judge. The detector is free, deterministic
and needs no calibration; where it and the judge disagree, one of them is wrong, and
those claims are listed because they are the useful place to look.

## Calibration workflow

```bash
python -m evals.harness label run_abc123       # worksheet with claims + their quotes
# fill in `supported` and `specific` in evals/labels/run_abc123.json
python -m evals.harness calibrate run_abc123   # confusion matrix, kappa, disagreements
```

The worksheet carries each claim alongside the exact quotes it rests on, so labelling
never requires going and finding the source. It is a file to fill in rather than an
interactive prompt, because labelling 50 claims is a half-hour job done in an editor
where you can revise an earlier call — which is also what makes self-consistency
checkable.

`calibrate` prints the confusion matrix, precision, recall, agreement, kappa and its
Landis & Koch band, plus every disagreement with the claim text. The disagreements are
the point: they are what you read to work out whether the judge prompt is wrong or your
own labelling was inconsistent.

## Golden set

`golden/*.json`, one target each. Current coverage: two companies, one industry, two
adversarial (an ambiguous name, a private company not in the SEC index).

`expected_points` — the 5–8 findings a competent analyst would reach for each target —
are **not yet written**, which is why `golden_coverage` reports `n/a (n=0)` rather than
a flattering 100%. Writing them is the highest-leverage unfinished work in the project:
it is what converts "the output looks good" into a measurement.

## Human labels

`labels/*.json` are committed on purpose. The judge is replaceable; the hand labels are
not, and committing them is what makes a calibration figure reproducible by someone
else. They must never be generated programmatically — a synthetic label teaches the
judge nothing and quietly destroys the only independent signal in the setup.
