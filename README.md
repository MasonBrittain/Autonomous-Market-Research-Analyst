# Autonomous Market Research Analyst

Give it a company name, a ticker, or an industry. It gathers evidence on its own,
builds a structured analytical brief, then **attacks its own conclusions** and throws
out the ones that do not survive.

Every claim in the output footnotes the exact sentence it came from. Claims that
cannot be traced to evidence are deleted, not softened.

```bash
analyst research "Thermo Fisher Scientific"
```

```
-> Resolve   ok  Thermo Fisher Scientific Inc. (TMO) conf=0.97
-> Scout     ok  34 docs, 3 round(s), coverage 86%, 7 tool calls
-> Librarian ok  61 facts from 22 usable docs (9 dup, 2 stub, 1 off-topic), quote integrity 100%
-> Analyst   ok  23 claims, 14 risks assessed, 2 dropped uncited, 0 invalid citations
-> Adversary ok  15 accepted, 4 revised, 4 rejected (17%), 3 contradictions
-> Scribe    ok  19 claims published
```

---

## Why this is not another SWOT generator

Ask a language model for a SWOT analysis and you will always get one. Roughly a third
of it will be uncited, miscategorised, or filler that would be equally true of any
company in the index. Three things here are aimed squarely at that.

**An adversary that rejects its own team's work.** Every claim is judged on four
independent axes by a separate pass: is it *supported* by the facts it cites (not
merely plausible), is it in the *correct category*, is it *specific* enough to be
wrong, and is the evidence *stale*? Verdicts are accept, revise, or reject. The
rejection rate is printed as a headline metric, because a review pass that never
rejects anything is not reviewing.

**The adversary also hunts for evidence against the claim.** News genuinely
disagrees — the same layoff is cost discipline in one outlet and distress in another.
Rather than silently picking a side, contradictions are surfaced in the brief under
the claim they undercut.

**Stated risks vs. observed reality.** A 10-K's Item 1A is a company's own ranked
list of what could go wrong, written under legal liability. Each disclosed risk is
cross-referenced against recent evidence and marked *materializing*, *quiet*, or
*contradicted*. It turns legal boilerplate into a live risk register, and it costs
nothing because EDGAR is free.

| Disclosed risk | Status | Evidence |
| --- | --- | --- |
| Manufacturing is concentrated with few contract partners in one region. | **Materializing** | [3] Two of three named suppliers disclosed capacity constraints this quarter. |
| Larger rivals may compress pricing. | **Contradicted** | [1] March price increases held with no volume loss. |

---

## The agents, and why each one exists

Work is divided by *where model judgement earns its cost*, not by processing stage.
Only one of these is an autonomous loop; making the others autonomous would add risk
without adding quality.

| Role | Does | Shape |
| --- | --- | --- |
| **Scout** | Resolves the entity, then gathers evidence in rounds until coverage saturates — deciding what it still does not know and going to get it | Bounded agent loop |
| **Librarian** | Strips boilerplate, rejects paywalls, deduplicates syndication, extracts atomic dated facts with verbatim quotes | Deterministic + parallel calls |
| **Analyst** | Builds the structured view: SWOT, risk cross-reference, catalysts, competitive position, open questions | Parallel constrained calls |
| **Adversary** | Attacks every claim; scans for contradicting evidence; rejects or repairs | Constrained call per claim |
| **Scribe** | Renders the brief from validated claims | Jinja2 + one call for the summary |

Three design choices do most of the work:

**Scout never sees article bodies.** Its tools return titles, dates, publishers and a
short gist. A retrieval decision does not need the full text, and feeding it in would
multiply the loop's cost by two orders of magnitude. Scout stops on *saturation*, not
on a fixed fetch count, and is explicitly told that "we have enough" is a correct
answer — bounded further by hard ceilings on rounds and tool calls.

**A quote that is not a literal substring of its source is not a fact.** The Librarian
requires a `verbatim_quote` with every extracted fact and verifies it character-for-
character against the source document (forgiving whitespace, nothing else). Fail the
check and the fact is dropped, not flagged. `quote_integrity` is reported on every run.

**The Scribe barely uses the model.** Sections are rendered deterministically from
claims that already survived review. The model writes only the executive summary, from
the validated claim list. Two runs over the same evidence produce byte-identical
reports apart from that summary — which is what makes eval diffs readable.

---

## What it measures about itself

Every brief ends with a provenance block, and a separate harness scores stored runs.
Integrity metrics should be 1.00; a drop is a bug, not a quality dip.

| Metric | Meaning |
| --- | --- |
| `quote_integrity` | Share of facts whose quote was genuinely in the source |
| `citation_validity` | Share of published claims citing at least one real fact |
| `dangling_citations` | Claims citing a fact that is not in the pack (must be 0) |
| `rejection_rate` | Share of claims the Adversary threw out |
| `coverage_score` | Share of Scout's seven dimensions evidenced |
| `dedup_rate` | Share of gathered documents found to be duplicates |
| `cost_usd` | Real spend, accumulated per node from `usage` on every call |

Judged metrics sit alongside those, and are gated: until a run's claims have been
hand-labelled, `citation_precision` and `specificity` report `trust: uncalibrated` and
`quotable: false`, with the blocking reason printed next to the figure. Every judged
proportion carries a Wilson interval rather than a bare percentage, and agreement with
human labels is reported as Cohen's kappa — because a judge that rubber-stamps
everything scores 0.90 raw agreement against a 90%-positive set while doing no work.

```bash
python -m evals.harness report              # structural scoreboard; non-zero exit on an integrity failure
python -m evals.harness judge <run_id>      # judged metrics (--live for the real model)
python -m evals.harness label <run_id>      # export a human-labelling worksheet
python -m evals.harness calibrate <run_id>  # judge vs. human labels: kappa, disagreements
```

Full detail, including why citation precision measures *residual* error rather than
error in the raw analysis, is in [`evals/README.md`](evals/README.md).

---

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set two things in `.env`:

```ini
ANTHROPIC_API_KEY=sk-ant-...
# SEC returns 403 on any request whose User-Agent lacks a contact email.
SEC_USER_AGENT=autonomous-market-analyst/0.1 (you@example.com)
```

```bash
analyst research "Apple"                  # full brief -> out/
analyst research "Apple" --stub           # no API key needed; real sources, placeholder analysis
analyst research "Apple" --show           # print the brief to stdout
analyst resolve "Delta"                   # entity resolution only
analyst runs                              # stored runs
analyst show <run_id> --format stats      # metrics for one run
```

`--stub` runs the whole pipeline with an offline stand-in for the model. It fetches
real documents and renders a real report with real citations and real metrics; only
the analytical text is placeholder. It is the fastest way to see the shape of the
output, and it is how the test suite covers the pipeline end to end.

---

## Sources

Free and sanctioned, by design:

- **SEC EDGAR** — 10-K Item 1A (risk factors) and Item 7 (MD&A), recent 8-Ks. The
  highest-value source here and the one comparable tools skip.
- **News RSS** — Google News and Yahoo Finance feeds, the publisher-sanctioned
  interface. Expect roughly a third of linked bodies to be paywalled; the RSS summary
  is retained as a dated, attributable fallback.
- **yfinance** — financial snapshot and peer metrics.
- **Peer discovery** — derived from the SEC's own SIC industry classification, since
  no free API offers a reliable peer list.

Every request goes through one `Fetcher` that honours `robots.txt`, rate-limits per
host, sends a descriptive User-Agent, and caches every response to disk by URL. That
cache is why the analysis stages can be re-tuned hundreds of times without re-fetching
a single document.

---

## Layout

```
src/analyst/
  models.py          one state object; checkpointed after every node
  config.py          process settings
  cache.py           content-addressed fetch cache
  prompts.py         versioned prompt loader
  tools/             fetch, entity resolution, EDGAR, news, financials, text
  scout/             the autonomous retrieval loop
  librarian/         triage, fact extraction, simhash deduplication
  analyst/           section construction
  adversary/         the red team
  scribe/            rendering + templates
  llm/               Claude client, strict schemas, offline stub
  orchestrator/      DAG runner + SQLite checkpoint store
prompts/v1/          every prompt, versioned and fingerprinted into each report
evals/
  harness.py         structural metrics + CLI
  judge.py           LLM-as-judge for the semantic metrics
  calibration.py     confusion matrix, Cohen's kappa, Wilson intervals, trust gate
  specificity.py     mechanical filler detector, checked against the judge
  labels/            hand labels (committed; the scarce artifact here)
  golden/            research targets and their expected findings
```

`simhash` deduplication is implemented directly rather than pulled from a dependency,
and the threshold is calibrated against measurement, not taste: identical copies
measure 0, syndicated copies (different chrome, trimmed tail) measure 1–4,
independently rewritten coverage of the same event measures ~33. The default of 8
catches syndication while correctly treating a rewrite as separate corroboration.

---

## Crash resume

The whole run is serialized to SQLite after every node, so a killed run resumes without
re-fetching or re-paying for completed work. The fact pack is rebuilt deterministically
— pinned to the run's creation time rather than wall clock — so recency ranking and the
cached prompt prefix are identical on resume.

```bash
analyst research "Apple"                  # dies during analysis
analyst research "Apple" --resume run_abc # picks up at analysis; zero re-fetches
```

Covered by a test that crashes the pipeline mid-analysis, reloads from the checkpoint,
finishes the run, and asserts the fetch count and cost ledger did not move.

---

## Cost

One model (`claude-opus-5`) at every stage, with per-node `effort` as the tuning lever
rather than swapping in cheaper models. Cost control comes from three places:

1. The fetch cache — iterate on analysis for free.
2. Prompt caching — the fact pack is the cached prefix shared by all eight Analyst
   calls and every Adversary judgement. The stable content sits first in the system
   prompt and only the per-section instruction varies after the breakpoint.
3. Doing cheap work first — paywall rejection and deduplication run before any model
   call, so we never pay to extract facts from the eighth copy of a wire story.

`cache_read_tokens` is reported on every run as the proof that (2) is working.

---

## Status

Working and tested end to end:

- All six pipeline stages, verified against live SEC and news data
- Structural eval harness (quote integrity, citation validity, coverage, dedup,
  rejection rate) running in CI on every commit
- Judged eval layer: LLM-as-judge for citation precision, golden coverage and
  specificity, with calibration against human labels — confusion matrix, Cohen's
  kappa, Wilson intervals, and a gate that refuses to call a judged number a result
  until the judge has been verified
- Crash resume and per-node cost accounting
- Entity resolution that refuses to guess: ambiguous names and private companies stop
  the run and list candidates rather than producing a confident brief about the wrong
  company
- 155 tests, `ruff` and `mypy` clean, no network or API key required

Not yet done, in priority order:

1. **Hand-write `expected_points` for the golden set.** 15–20 targets × 5–8 findings.
   The harness that consumes them is built and gated; the labels themselves are human
   work, and they are what turn "the output looks good" into a number.
2. **Label ~50 claims and calibrate the judge.** The machinery is in place —
   `evals.harness label` produces a worksheet carrying each claim with its quotes — but
   until the labels exist, judged metrics correctly refuse to report themselves as
   results.
3. **Verify prompt caching against the live API.** The layout is implemented and its
   structure is unit-tested, but `cache_read_tokens > 0` can only be confirmed with a
   real key.
4. FastAPI service, worker queue, and Azure deployment.

---

## Not investment advice

This produces research notes from public sources. It can be incomplete or wrong. Every
claim is footnoted specifically so it can be checked rather than trusted.
