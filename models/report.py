"""Structured outputs of the analysis stages and the final report.

Everything the UI renders comes from these models. Anything a user could read as a factual claim
carries ``evidence_ids`` that resolve to items in the evidence store.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field

# -------------------------------------------------------------------------- market

class MarketSignal(BaseModel):
    """Demand for one skill within the sampled listings. ``total`` is the real sample size."""

    skill: str
    count: int = Field(ge=0)
    total: int = Field(ge=0)
    scope: Literal["in_listing", "adjacent"]  # named by this listing, or common in the sample but not named
    missing_from_resume: bool | None = None  # ``None`` when no resume was supplied
    statement: str  # FACT
    interpretation: str  # neutral INTERPRETATION
    action: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def percent(self) -> float:
        return round(100 * self.count / self.total, 1) if self.total else 0.0


class MarketAnalysis(BaseModel):
    role_query: str
    location: str | None = None
    sample_size: int = 0
    low_sample: bool = False
    excluded_off_role: int = 0
    signals: list[MarketSignal] = Field(default_factory=list)
    method: str = ""
    search_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------- news

Relevance = Literal["high", "medium", "low"]


class NewsHighlight(BaseModel):
    """A retained news item. ``summary`` is verbatim source text (FACT); ``topic`` and
    ``relevance_reason`` are ScoutLens's neutral INTERPRETATION."""

    title: str
    url: str
    publisher: str | None = None
    date_text: str | None = None
    published_at: datetime | None = None
    summary: str
    topic: str
    relevance: Relevance
    relevance_reason: str
    evidence_id: str
    tier: Literal[1, 2, 3]


class NewsSignal(BaseModel):
    company: str
    items: list[NewsHighlight] = Field(default_factory=list)
    total_found: int = 0
    excluded_irrelevant: int = 0
    excluded_old: int = 0
    search_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------- hiring signals

ComparisonKind = Literal["volume", "listing", "skills", "experience", "location", "work_mode", "salary"]


class SkillFrequency(BaseModel):
    """How many of the analysed listings mention a skill. ``total`` is the real sample size."""

    skill: str
    count: int = Field(ge=0)
    total: int = Field(ge=0)
    in_original: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def percent(self) -> float:
        return round(100 * self.count / self.total, 1) if self.total else 0.0


class ExperienceObservation(BaseModel):
    title: str
    location: str | None = None
    raw: str
    min_years: float | None = None
    max_years: float | None = None
    evidence_id: str | None = None


class LocationCount(BaseModel):
    location: str
    count: int


class SignalComparison(BaseModel):
    """One neutral observation comparing the original listing with the employer's other listings."""

    kind: ComparisonKind
    fact: str
    interpretation: str = ""
    action: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    strength: float = 0.0  # internal ranking hint, never shown as a score


class HiringSignal(BaseModel):
    company: str
    sample_size: int = Field(ge=0, description="Other current listings from this employer that were analysed")
    excluded_other_company: int = 0
    similar_count: int = 0
    original_found: bool = False
    original_via: str | None = None

    skill_frequencies: list[SkillFrequency] = Field(default_factory=list)
    experience: list[ExperienceObservation] = Field(default_factory=list)
    locations: list[LocationCount] = Field(default_factory=list)
    work_modes: dict[str, int] = Field(default_factory=dict)
    salary_listed_count: int = 0
    salary_examples: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)

    comparisons: list[SignalComparison] = Field(default_factory=list)
    listing_evidence_ids: list[str] = Field(default_factory=list)
    search_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return self.sample_size > 0
