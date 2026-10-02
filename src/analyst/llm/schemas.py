"""Request/response schemas for every model call in the pipeline.

These are deliberately flat and primitive-typed. They are the contract the model
must satisfy, and a shallow schema is both cheaper to send and far less likely to
be rejected or half-filled than a nested one. Dates are strings here and parsed
on our side -- a model that cannot produce a date returns null instead of
inventing one.

Internal `models.py` types stay richer; these are the wire format.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Scout
# --------------------------------------------------------------------------- #


class ScoutPlan(BaseModel):
    """Opening retrieval plan: what to search for and why."""

    queries: list[str] = Field(description="3-8 news search queries, no boolean operators")
    dimensions_targeted: list[str] = Field(description="Coverage dimensions these queries address")
    rationale: str


class CoverageVerdict(BaseModel):
    """Scout's self-assessment after a gathering round.

    `saturated` is the stopping condition: more fetching would not change the
    analysis. Scout is expected to say so rather than spend its whole budget.
    """

    covered: list[str] = Field(description="Dimension names with sufficient evidence")
    gaps: list[str] = Field(description="Dimension names still lacking evidence")
    next_queries: list[str] = Field(description="Queries to close the gaps; empty if saturated")
    saturated: bool
    reasoning: str


# --------------------------------------------------------------------------- #
# Librarian
# --------------------------------------------------------------------------- #


class RelevanceVerdict(BaseModel):
    relevant: bool
    reason: str
    dimensions: list[str] = Field(description="Dimensions this document provides evidence for")


class ExtractedFact(BaseModel):
    text: str = Field(description="One atomic, self-contained factual statement")
    verbatim_quote: str = Field(description="Exact substring of the source supporting the text")
    happened_at: str | None = Field(description="ISO date (YYYY-MM-DD) or null if undated")
    polarity: str = Field(description="positive | negative | neutral")
    dimensions: list[str]
    salience: float = Field(description="0.0-1.0 importance to an investor or analyst")


class FactExtraction(BaseModel):
    facts: list[ExtractedFact]


class DocumentTriage(BaseModel):
    """Relevance and extraction in one call -- two calls per document would double
    cost for no quality gain, since both need the same document in context."""

    relevant: bool
    reason: str
    dimensions: list[str]
    facts: list[ExtractedFact]


# --------------------------------------------------------------------------- #
# Analyst
# --------------------------------------------------------------------------- #


class DraftClaim(BaseModel):
    statement: str = Field(description="One specific, falsifiable analytical claim")
    rationale: str = Field(description="Why the cited facts support this claim")
    confidence: float = Field(description="0.0-1.0")
    fact_ids: list[str] = Field(description="IDs of supporting facts, from the fact pack")


class ClaimSet(BaseModel):
    claims: list[DraftClaim]
    insufficient_evidence: bool = Field(
        description="True if the fact pack cannot support any claim here"
    )
    note: str = Field(description="Empty unless insufficient_evidence is true")


class RiskAssessment(BaseModel):
    risk_index: int = Field(description="Index of the risk in the provided list")
    summary: str = Field(description="The disclosed risk in one plain sentence")
    status: str = Field(description="materializing | quiet | contradicted")
    fact_ids: list[str]
    reasoning: str


class RiskAssessmentSet(BaseModel):
    assessments: list[RiskAssessment]


class OpenQuestionSet(BaseModel):
    questions: list[str] = Field(
        description="What the evidence could not answer. Honest gaps, not rhetorical questions."
    )


# --------------------------------------------------------------------------- #
# Adversary
# --------------------------------------------------------------------------- #


class ClaimJudgement(BaseModel):
    """The red-team verdict on a single claim."""

    supported: bool = Field(description="Do the cited facts actually establish the claim?")
    correct_section: bool = Field(
        description="Right category? Strengths/Weaknesses are internal and present; "
        "Opportunities/Threats are external and forward-looking."
    )
    specific: bool = Field(description="False for generic filler that would fit any company")
    stale: bool = Field(description="True if the supporting evidence is outside the window")
    contradicting_fact_ids: list[str] = Field(
        description="Facts in the pack that cut against this claim"
    )
    verdict: str = Field(description="accept | revise | reject")
    reasoning: str


class RevisedClaim(BaseModel):
    statement: str
    rationale: str
    confidence: float
    fact_ids: list[str]


# --------------------------------------------------------------------------- #
# Scribe
# --------------------------------------------------------------------------- #


class ExecutiveSummary(BaseModel):
    """The only free-text the model writes into the report.

    Everything else is rendered deterministically from validated claims, so this
    is the only place a hallucination could reach the page -- and it is written
    from claims that already survived the Adversary.
    """

    summary: str = Field(description="3-5 sentences, no claim absent from the provided set")
    headline: str = Field(description="One line, under 90 characters")
