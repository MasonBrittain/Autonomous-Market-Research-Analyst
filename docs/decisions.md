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

## 13. A database-backed queue, not a message broker

**Decided:** 2026-10-06

Jobs live in the same SQLite database as the runs they produce. Two guarantees depend
on that:

- Creating a run and queueing its job happen in one transaction. A crash cannot leave
  a job pointing at a run that was never written, or a run nobody will ever process.
- A worker's checkpoint is fenced against its lease in the same statement that writes
  it (`UPDATE runs ... WHERE EXISTS (SELECT 1 FROM jobs ...)`).

Neither is possible across a broker and a database without distributed transactions.

**Rejected:** Redis or a cloud queue service. Either is another moving part to run,
and either loses the two guarantees above.

**Consequence:** throughput is bounded by one database's write rate. A research run
takes minutes, so the queue sees a few writes per minute per worker. That is nowhere
near any limit that matters here.

---

## 14. At-least-once delivery, made safe by resume and fencing

A claim is a lease, renewed by heartbeats. A dead worker's job is claimed again after
its lease lapses, so delivery is at-least-once, and the design has to make a second
delivery harmless.

Two mechanisms make it harmless. The pipeline resumes from its last checkpoint, so
the second worker skips every node the first one finished. And every claim increments
`attempts`, which is used as a fencing token. Lease renewals, completions and
checkpoint writes all carry the token, and all are refused once another worker has
claimed the job. A stalled worker that wakes up after its lease lapsed (a long GC
pause, a hung network call) finds out at its next write and stops. It cannot overwrite
the worker that replaced it.

A mutation check confirms the fence is load-bearing: with the `EXISTS` clause removed
from the checkpoint write, both split-brain tests fail.

---

## 15. The heartbeat runs on its own thread

The pipeline's model calls are synchronous and hold the event loop for seconds at a
time. An asyncio heartbeat would starve during those calls, the lease would lapse
mid-run, and a second worker would start a job that was still being worked on.

The heartbeat therefore runs on a dedicated thread with its own database connection.
A real-time test blocks the main thread for 2.5 seconds against a 1-second lease and
checks that no rival can claim the job. A control test shows the same lease does lapse
without the heartbeat, so the first test can actually fail.

The worker also refuses a lease shorter than three heartbeats. A transient database
error costs one beat, not the job.

---

## 16. Resolve the company before queueing

`POST /research` resolves the query against the SEC index inside the request, before
anything is queued. That buys two things:

- An ambiguous query ("Delta") is rejected immediately with its candidates, rather
  than becoming a job that fails minutes later with nobody watching.
- Deduplication keys on the resolved company, so "AAPL" and "Apple Inc." are one
  paid run rather than two.

The cost is a dependency on the SEC index at request time. The index is cached in
memory for a day, and an outage is reported as a 503 rather than a 500.

---

## 17. What counts as "the same research"

Two requests are the same work when they resolve to the same company and would run
under the same configuration. The configuration fingerprint includes a hash of the
prompt text. A brief produced before a prompt edit is never served as if it came from
the new prompts, and an offline stub brief never answers a request for a real one.

A matching brief that finished within the TTL (default 24 hours) is returned with a
200 instead of re-run. A matching brief already in progress is joined instead of
duplicated. `force: true` bypasses the first rule but not the second: starting a second
copy of work already in flight is never useful.

---

## 18. Coalescing is enforced by the schema

At most one job per (company, configuration) may be in flight. This is a partial
unique index (`... WHERE status IN ('queued', 'running')`), not check-then-insert in
application code.

On SQLite, `BEGIN IMMEDIATE` already serialises enqueues, so the index is a backstop.
On a database without a database-wide write lock, it is the guarantee. A test fires
eight simultaneous requests for the same company and checks that exactly one job and
one run exist afterwards.

---

## 19. Reads open, writes behind a key

When `ANALYST_SERVICE_API_KEY` is set, starting and cancelling work need a bearer
token, compared in constant time. Reads stay open: they cost nothing, and a brief is
meant to be shared.

`analyst serve` warns when it binds beyond localhost without a key, and the compose
file publishes the port on 127.0.0.1 only. The default fails safe.

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


**Resume silently dropped two report sections.** Risk factors (produced by Scout,
consumed by the Analyst) and the SEC index (loaded by resolve, needed by Scout for
peers) were held on the `Pipeline` object, not on the checkpointed run. A run resumed
in a fresh process lost the stated-risks section and its peer set, without any error.
The existing crash-resume test checked that a report came out, not that it was
complete. Both now live where checkpoints can see them. Regression tests resume from
each boundary and compare against a clean run.

**An installed package could not find its prompts.** Prompts lived at the repository
root and were located through `Path(__file__).parents[2]`. That only resolves from a
source checkout. From a built wheel, and so from any container image, every model call
failed with `PromptNotFound`. Prompts now ship inside the package. A CI job installs
the built wheel outside the source tree and loads every prompt, template and static
file from it.

**Filers style headings letter by letter.** Microsoft's FY2026 10-K reaches us as
`ITEM 1A. RIS K FACTORS`: the heading wraps single letters in their own spans, and
flattening HTML turns each tag boundary into a space. The strict heading pattern
matched only the table-of-contents entry, which is too short to count, so the
stated-risks section vanished for that filer. Heading patterns now tolerate whitespace
inside words. Microsoft went from 0 extracted risk factors to 25, and Apple was
unchanged.

**Paywalled pages were counted twice.** The stub filter and the model's relevance
check both set `relevant = False`, and the off-topic count read that flag afterwards.
A live run reported "25 stub, 25 off-topic" out of 43 documents. The true off-topic
count was 0. Off-topic is now counted where the model makes the call.

**Most Google News bodies are unreachable.** On a live Microsoft run, 25 of 44
gathered documents were rejected as stubs. That is higher than the roughly one third
estimated in decision 9. Google News RSS links are redirect pages that need JavaScript
to resolve, so body fetches usually land on an interstitial. The RSS summary survives
as evidence, but this caps news depth. Decoding those redirects, or adding a source
with direct article links, is the obvious next improvement to evidence quality.
