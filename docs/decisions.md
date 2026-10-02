# Design decisions

Short records of choices that were genuinely contested, so the reasoning survives.

---

## 1. Scout plans and uses fixed tools; it does not write scrapers

**Decided:** 2026-10-01

An earlier sketch had the first agent write a web-scraping script at runtime and
execute it. Rejected for three reasons:

- It is arbitrary code execution inside our own process.
- It is non-reproducible. A different scraper each run means you cannot tell a model
  regression from a site layout change — which destroys the eval harness before it is
  built.
- There is little to discover. News sites publish RSS and JSON-LD; the structured
  sources (EDGAR, yfinance) have documented APIs.

Scout instead plans retrieval — picking sources, writing queries, assessing coverage —
over fixed, individually tested adapters.

**Consequence:** no scraper-generation showcase. Acceptable: the autonomy that matters
is the coverage-driven loop, and that survives intact.

---

## 2. Scout sees metadata only

A retrieval decision needs a title, a date, a publisher and a gist. Giving the loop
full article bodies would multiply its context and cost by two orders of magnitude for
no improvement in what it decides to fetch next.

`Evidence.digest()` is the enforced boundary. Full text goes to the store for fact
extraction and quote verification; Scout never receives it.

---

## 3. Saturation, not a fetch count, is the stopping condition

Scout is asked after each round which of seven coverage dimensions are evidenced and
whether more searching would change the analysis. It is told explicitly that "we have
enough" is a correct answer.

Backstops, because an agent that never converges is worse than one that stops early:
hard ceilings on rounds and tool calls, and a forced stop when a round returns no new
documents.

**Rejected alternative:** a fixed fetch list. Cheaper and more predictable, but then
nothing about the system is autonomous and coverage cannot adapt to a thinly covered
company.

---

## 4. The Adversary is a separate pass, and its rejection rate is published

An Analyst asked for a SWOT always produces one. Without a check, a large share of the
output is uncited, miscategorised, or filler.

Four independent axes — supported / correct section / specific / stale — then accept,
revise, or reject. One repair attempt per flagged claim; if a narrowed restatement
still fails, the claim was not supportable.

Publishing `rejection_rate` is deliberate. A review pass that never rejects anything
looks like it is working and is not, and the only way to notice is to watch the number.

**Also deliberate:** the Adversary scans the whole fact pack for evidence *against* the
claim, and contradictions are surfaced in the brief rather than resolved. Disagreement
between sources is information, not noise to be smoothed away.

---

## 5. An unverifiable quote is not a fact

Every extracted fact carries a `verbatim_quote` that must appear character-for-character
in its source document (whitespace differences forgiven, since those are an artefact of
HTML extraction). Failing facts are **dropped**, not flagged.

Everything downstream — the Adversary's support check, the report's footnotes — assumes
a quote can be found in the source. A fact that breaks that assumption is worse than a
missing one.

`quote_integrity` is reported per run. On live data it has been observed below 1.0,
which means the gate is doing real work rather than decorating the pipeline.

---

## 6. The Scribe renders; it does not write

Sections are built with Jinja2 from claims that already survived review. The model
writes only the executive summary, and only from the validated claim list.

This confines the last place a hallucination could reach the page to summarising
statements that are already cited, and it makes rendering deterministic — two runs over
the same evidence differ only in that summary, which is what makes eval diffs readable.

---

## 7. One model, effort as the lever

`claude-opus-5` at every stage, tuned per node with `output_config.effort` rather than
swapping in cheaper models for the high-volume stages.

The reasoning: establish the quality ceiling first, then test cheaper models per node
*against the eval scoreboard* and keep the downgrade only where quality holds. Tiering
first would mean never learning what the cheaper extraction cost.

Cost control therefore comes from caching and from ordering cheap work first, not from
model choice. Paywall rejection and deduplication run before any model call, so we never
pay to extract facts from the eighth copy of a wire story.

---

## 8. Caching layout: stable content first, instruction last

Prompt caching is a prefix match, so:

```
system = [ shared role text , fact pack  <- cache breakpoint ]
user   = per-section instruction + inputs
```

Because the role text and fact pack are byte-identical across all eight Analyst calls
and every Adversary judgement, each call after the first reads the pack from cache. Had
the section instruction lived in the system prompt, every section would start a fresh
prefix and the pack would be billed eight times.

The fact pack is rebuilt from `run.created_at` rather than wall-clock time precisely so
that a resumed run produces a byte-identical prefix.

**Unverified:** `cache_read_tokens > 0` needs a live API key to confirm. The structure
is unit-tested; the behaviour is not yet.

---

## 9. Free sources only, with EDGAR carrying the weight

RSS over scraping: publisher-sanctioned, clean metadata, no ToS exposure. The cost is
recall — roughly a third of linked article bodies are paywalled.

That trade is acceptable because **EDGAR is the better source anyway.** Item 1A is
management's own ranked list of threats and Item 7 is management explaining its own
numbers, both under legal liability. A paid search API drops in later behind the same
adapter interface.

---

## 10. Hand-rolled orchestrator

Six nodes, linear. Owning the loop is what makes crash-resume, per-node cost accounting,
and a typed error chain straightforward rather than framework-mediated.

**Rejected:** LangGraph. It would add a dependency and a layer of indirection to solve a
problem this DAG does not have.

---

## 11. Resolution refuses to guess

A query resolving below 0.6 confidence, or to no ticker and no CIK, stops the run and
returns candidates.

Entity confidence is computed from *separation*, not absolute similarity: two companies
scoring 0.95 and 0.94 is exactly the ambiguous case that must not be resolved silently.
A word-boundary prefix pass supplements edit distance, because "delta" vs "delta air
lines" scores only 0.53 on `SequenceMatcher` — without it, a genuinely ambiguous query
would surface no candidates at all and the run could not say what it was torn between.

A confident brief about the wrong company is the worst output this system can produce.

---

## 12. Integrity metrics only apply to completed runs

A run that correctly stopped — ambiguous target, fetch failure — has no claims to check.
Scoring it zero would make the scoreboard show a regression when the pipeline behaved
exactly as designed. Such runs report `n/a` and are excluded from the integrity gate.

---

## Operational findings

**SEC requires a contact email in the User-Agent.** Verified directly: a descriptive UA
without an address returns 403 on every endpoint; the same UA with one returns 200.
Validated at startup so this surfaces as an actionable message rather than a confusing
mid-pipeline failure.

**SEC filings are full of inline XBRL.** Flattened to text, an 8-K primary document
opens with long runs of identifiers — `aapl-20260730 false 0000320193
us-gaap:CommonStockMember ...` — which read as on-topic to a relevance check while
containing no facts. Stripped by line-level identifier density before extraction.

**A 10-K names "Item 1A. Risk Factors" at least twice**, once in the table of contents.
Extraction takes the longest plausible span between a start marker and a following end
marker, not the first match.
