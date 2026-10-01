"""The single state object every stage reads and writes.

Three properties are load-bearing and worth preserving if this file is edited:

1. Everything is JSON-serializable, so a run can be checkpointed after each node
   and resumed after a crash without re-fetching or re-paying.
2. `Evidence` separates the full body (`clean_text`, stored) from `digest()`
   (metadata only, shown to Scout). Scout never sees article bodies, which is
   what keeps an autonomous retrieval loop affordable and bounded.
3. Every `Claim` carries `evidence_ids`. A claim that cannot name its evidence is
   a bug, not a style problem -- the Adversary stage enforces this.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def content_hash(text: str) -> str:
    """Stable hash of document text, used for exact-duplicate detection."""
    return hashlib.sha256(text.strip().encode("utf-8", "ignore")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class SourceType(str, Enum):
    NEWS = "news"
    FILING_10K = "filing_10k"
    FILING_8K = "filing_8k"
    FINANCIALS = "financials"
    PEERS = "peers"
    WEB = "web"


class Dimension(str, Enum):
    """Scout's coverage checklist.

    Scout's stopping condition is saturation across these dimensions rather than
    a fixed number of fetches -- this is what makes retrieval autonomous instead
    of scripted.
    """

    FINANCIAL_HEALTH = "financial_health"
    COMPETITIVE_POSITION = "competitive_position"
    RECENT_EVENTS = "recent_events"
    REGULATORY_LEGAL = "regulatory_legal"
    LEADERSHIP = "leadership"
    PRODUCT_TECH = "product_tech"
    MARKET_TRENDS = "market_trends"

    @classmethod
    def all(cls) -> list[Dimension]:
        return list(cls)


class Section(str, Enum):
    SNAPSHOT = "snapshot"
    SWOT = "swot"
    STATED_RISKS = "stated_risks"
    CATALYSTS = "catalysts"
    COMPETITIVE = "competitive"
    OPEN_QUESTIONS = "open_questions"


class Quadrant(str, Enum):
    STRENGTH = "strength"
    WEAKNESS = "weakness"
    OPPORTUNITY = "opportunity"
    THREAT = "threat"


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RiskStatus(str, Enum):
    """Cross-reference of a company's own stated risk against observed reality."""

    MATERIALIZING = "materializing"
    QUIET = "quiet"
    CONTRADICTED = "contradicted"


class Verdict(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


class RunStatus(str, Enum):
    PENDING = "pending"
    RESOLVING = "resolving"
    SCOUTING = "scouting"
    CURATING = "curating"
    ANALYZING = "analyzing"
    CHALLENGING = "challenging"
    COMPOSING = "composing"
    DONE = "done"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# Entity resolution
# --------------------------------------------------------------------------- #


class EntityCandidate(BaseModel):
    name: str
    ticker: str | None = None
    cik: str | None = None
    score: float = 0.0


class Entity(BaseModel):
    """A resolved research target.

    Resolution happens before anything else because every downstream relevance
    judgement depends on knowing whether "Apple" means Apple Inc., the fruit, or
    the record label.
    """

    query: str
    name: str
    ticker: str | None = None
    cik: str | None = None
    aliases: list[str] = Field(default_factory=list)
    sic: str | None = None
    industry: str | None = None
    exchange: str | None = None
    peers: list[str] = Field(default_factory=list)
    is_industry: bool = False
    confidence: float = 0.0
    candidates: list[EntityCandidate] = Field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        """Several candidates scored close together -- resolving silently would be a coin flip."""
        return self.confidence < 0.6 and len(self.candidates) > 1

    @property
    def needs_clarification(self) -> bool:
        """True when the run should stop and ask instead of researching a guess.

        Covers both failure modes: genuine ambiguity (several close candidates) and
        no usable match at all, which would otherwise proceed with a bare query
        string as the company name.
        """
        if self.is_industry:
            return False
        return self.confidence < 0.6 or not (self.ticker or self.cik)

    def search_terms(self) -> list[str]:
        terms = [self.name, *self.aliases]
        if self.ticker:
            terms.append(self.ticker)
        seen: set[str] = set()
        out: list[str] = []
        for t in terms:
            key = t.lower().strip()
            if key and key not in seen:
                seen.add(key)
                out.append(t)
        return out


# --------------------------------------------------------------------------- #
# Evidence and facts
# --------------------------------------------------------------------------- #


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ev"))
    source_type: SourceType
    url: str
    title: str = ""
    publisher: str = ""
    published_at: datetime | None = None
    raw_text: str = ""
    clean_text: str = ""
    hash: str = ""
    gist: str = ""

    # Set by the Librarian.
    relevant: bool | None = None
    relevance_reason: str = ""
    dimensions: list[Dimension] = Field(default_factory=list)
    cluster_id: str | None = None
    duplicate_of: str | None = None
    authority: float = 0.5

    @property
    def is_usable(self) -> bool:
        return self.relevant is not False and self.duplicate_of is None

    def digest(self) -> dict[str, object]:
        """The metadata-only view handed to Scout.

        Scout decides what to gather next from titles, dates and gists -- never
        from article bodies. Returning bodies here would blow up the agent loop's
        context and cost for no decision-making benefit.
        """
        return {
            "id": self.id,
            "type": self.source_type.value,
            "title": self.title[:160],
            "publisher": self.publisher,
            "date": self.published_at.date().isoformat() if self.published_at else None,
            "gist": self.gist[:240],
            "words": len(self.clean_text.split()),
        }


class Fact(BaseModel):
    """An atomic, dated, quoted claim extracted from one piece of evidence.

    The verbatim quote is mandatory: it is what makes a downstream analytical
    claim checkable by the Adversary rather than merely plausible.
    """

    id: str = Field(default_factory=lambda: new_id("f"))
    evidence_id: str
    text: str
    verbatim_quote: str
    happened_at: date | None = None
    polarity: Polarity = Polarity.NEUTRAL
    dimensions: list[Dimension] = Field(default_factory=list)
    salience: float = 0.5


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


class AdversaryVerdict(BaseModel):
    verdict: Verdict
    supported: bool
    correct_section: bool
    specific: bool
    stale: bool = False
    contradicting_fact_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: new_id("c"))
    section: Section
    quadrant: Quadrant | None = None
    statement: str
    rationale: str = ""
    confidence: float = 0.5
    evidence_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    verdict: AdversaryVerdict | None = None
    revision_of: str | None = None

    @property
    def survived(self) -> bool:
        return self.verdict is None or self.verdict.verdict is not Verdict.REJECT


class StatedRisk(BaseModel):
    """One risk factor the company itself disclosed, scored against the news."""

    id: str = Field(default_factory=lambda: new_id("r"))
    risk_text: str
    summary: str
    status: RiskStatus = RiskStatus.QUIET
    fact_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""


class Snapshot(BaseModel):
    """Deterministic facts about the entity. No model involved."""

    as_of: date | None = None
    market_cap: float | None = None
    revenue_ttm: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    pe_ratio: float | None = None
    debt_to_equity: float | None = None
    free_cash_flow: float | None = None
    employees: int | None = None
    price: float | None = None
    price_change_52w: float | None = None
    extras: dict[str, object] = Field(default_factory=dict)


class PeerMetric(BaseModel):
    ticker: str
    name: str = ""
    market_cap: float | None = None
    revenue_ttm: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    pe_ratio: float | None = None


# --------------------------------------------------------------------------- #
# Coverage, cost, orchestration
# --------------------------------------------------------------------------- #


class CoverageAssessment(BaseModel):
    """Scout's self-assessment after a round of gathering."""

    round_number: int = 0
    covered: list[Dimension] = Field(default_factory=list)
    gaps: list[Dimension] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)
    saturated: bool = False
    reasoning: str = ""

    @property
    def score(self) -> float:
        total = len(Dimension.all())
        return round(len(set(self.covered)) / total, 3) if total else 0.0


# Per-million-token rates. Cache reads bill at roughly a tenth of the input
# rate and cache writes at roughly 1.25x; both are applied in `UsageRecord`.
MODEL_RATES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


class UsageRecord(BaseModel):
    node: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int = 0
    stop_reason: str | None = None

    @property
    def cost_usd(self) -> float:
        rate_in, rate_out = MODEL_RATES.get(self.model, (5.00, 25.00))
        per_token_in = rate_in / 1_000_000
        return round(
            self.input_tokens * per_token_in
            + self.cache_read_tokens * per_token_in * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * per_token_in * CACHE_WRITE_MULTIPLIER
            + self.output_tokens * (rate_out / 1_000_000),
            6,
        )


class CostLedger(BaseModel):
    records: list[UsageRecord] = Field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    @property
    def total_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 4)

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def cache_read_tokens(self) -> int:
        return sum(r.cache_read_tokens for r in self.records)

    def by_node(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.records:
            out[r.node] = round(out.get(r.node, 0.0) + r.cost_usd, 6)
        return out


class NodeState(BaseModel):
    status: NodeStatus = NodeStatus.PENDING
    attempts: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> int:
        if not (self.started_at and self.finished_at):
            return 0
        return int((self.finished_at - self.started_at).total_seconds() * 1000)


class RunConfig(BaseModel):
    """Snapshotted onto every run so an eval result is attributable to a config."""

    model: str = "claude-opus-5"
    effort: dict[str, str] = Field(
        default_factory=lambda: {
            "scout": "medium",
            "librarian": "low",
            "analyst": "high",
            "adversary": "high",
            "scribe": "medium",
        }
    )
    lookback_days: int = 120
    max_scout_rounds: int = 4
    max_scout_tool_calls: int = 24
    max_evidence: int = 80
    max_facts_in_pack: int = 120
    fact_pack_token_budget: int = 40_000
    min_body_words: int = 200
    # Max Hamming distance between 64-bit simhashes. Calibrated on real article
    # lengths: identical=0, syndicated copies=4, independently rewritten coverage
    # of the same event=33, unrelated=36. 8 catches syndication with wide margin
    # while correctly treating rewrites as distinct corroboration. Re-tune in evals.
    near_duplicate_threshold: int = 8
    allow_revision: bool = True
    stub: bool = False
    prompt_version: str = "v1"


class Report(BaseModel):
    markdown: str = ""
    html: str = ""
    executive_summary: str = ""
    generated_at: datetime = Field(default_factory=_utcnow)


class ResearchRun(BaseModel):
    """The run. One of these is checkpointed to SQLite after every node."""

    model_config = ConfigDict(validate_assignment=False)

    id: str = Field(default_factory=lambda: new_id("run"))
    query: str
    config: RunConfig = Field(default_factory=RunConfig)
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    entity: Entity | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    stated_risks: list[StatedRisk] = Field(default_factory=list)
    snapshot: Snapshot | None = None
    peers: list[PeerMetric] = Field(default_factory=list)
    coverage: list[CoverageAssessment] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    nodes: dict[str, NodeState] = Field(default_factory=dict)
    ledger: CostLedger = Field(default_factory=CostLedger)
    report: Report | None = None
    error: str | None = None

    # -- lookups ----------------------------------------------------------- #

    def evidence_by_id(self, eid: str) -> Evidence | None:
        return next((e for e in self.evidence if e.id == eid), None)

    def fact_by_id(self, fid: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fid), None)

    def usable_evidence(self) -> list[Evidence]:
        return [e for e in self.evidence if e.is_usable]

    def claims_in(self, section: Section) -> list[Claim]:
        return [c for c in self.claims if c.section is section and c.survived]

    def swot(self, quadrant: Quadrant) -> list[Claim]:
        return [c for c in self.claims_in(Section.SWOT) if c.quadrant is quadrant]

    def node(self, name: str) -> NodeState:
        return self.nodes.setdefault(name, NodeState())

    # -- metrics ----------------------------------------------------------- #

    @property
    def latest_coverage(self) -> CoverageAssessment | None:
        return self.coverage[-1] if self.coverage else None

    @property
    def dedup_rate(self) -> float:
        if not self.evidence:
            return 0.0
        dupes = sum(1 for e in self.evidence if e.duplicate_of is not None)
        return round(dupes / len(self.evidence), 3)

    @property
    def rejection_rate(self) -> float:
        judged = [c for c in self.claims if c.verdict is not None]
        if not judged:
            return 0.0
        rejected = sum(1 for c in judged if c.verdict and c.verdict.verdict is Verdict.REJECT)
        return round(rejected / len(judged), 3)

    def touch(self) -> None:
        self.updated_at = _utcnow()
