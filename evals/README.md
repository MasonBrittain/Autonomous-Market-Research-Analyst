# Eval harness

Two kinds of metric, kept deliberately separate.

## Structural (implemented, no API key, runs in CI)

Computed from a stored run with no model involved:

| Metric | Target | Meaning |
| --- | --- | --- |
| `citation_validity` | 1.00 | Every published claim cites at least one fact. Below 1.0 is a bug. |
| `quote_integrity` | 1.00 | Every fact's quote is a literal substring of its source. Below 1.0 is a bug. |
| `dangling_citations` | 0 | No claim cites a fact that is not in the pack. |
| `coverage_score` | tune | Share of Scout's seven dimensions evidenced. |
| `dedup_rate` | tune | Share of gathered documents found to be duplicates. |
| `rejection_rate` | > 0 | Share of claims the Adversary rejected. Zero means the pass is not working. |
| `cost_usd` | tune | Real spend per brief. |
| `cache_read_tokens` | > 0 | Proof the cached fact-pack prefix is being reused. |

```bash
python -m evals.harness golden          # list the golden set
python -m evals.harness score <run_id>  # score one stored run
python -m evals.harness report          # scoreboard across stored runs; non-zero exit on integrity failure
```

## Semantic (not implemented)

These need a judge and are deliberately unimplemented rather than faked:

- **citation_precision** -- does the cited evidence actually establish the claim?
- **golden_coverage** -- what share of each target's hand-written `expected_points` did the brief hit?
- **specificity** -- share of claims naming a number, date or party.

Before any of these produce a number worth quoting, the judge must be **calibrated**:
hand-label ~50 claims, then publish the judge's agreement rate with those labels. An
uncalibrated judge is a vibe with a decimal point.

## Golden set

`golden/*.json`, one target each. The `expected_points` lists are the highest-leverage
unfinished work in the project: 5-8 findings per target that a competent analyst would
hit. Writing them by hand is what turns "the output looks good" into a measurement.

Current coverage: 2 companies, 1 industry, 2 adversarial (ambiguous name, private
company). The plan calls for 15-20 targets including a recent IPO, a small-cap, and one
company with a scandal inside the window.
