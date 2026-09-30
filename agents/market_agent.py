"""Market Agent: samples current listings for the role and measures skill demand within that sample."""
from __future__ import annotations

import logging

from models.opportunity import OpportunityProfile
from models.plan import PlannedSearch
from models.report import MarketAnalysis
from models.resume import ResumeProfile
from services.analysis.market import analyze_market
from services.evidence.store import EvidenceStore
from services.serpapi.client import SerpApiClient
from services.serpapi.jobs import fetch_jobs
from utils.errors import SerpApiError

logger = logging.getLogger(__name__)


class MarketAgent:
    def __init__(self, client: SerpApiClient, store: EvidenceStore, default_location: str | None = None) -> None:
        self._client = client
        self._store = store
        self._default_location = default_location  # what SerpApi searches when the step names no location

    def investigate(
        self,
        opp: OpportunityProfile,
        step: PlannedSearch,
        *,
        resume: ResumeProfile | None = None,
        refresh: bool = False,
    ) -> MarketAnalysis:
        """Fatal SerpApi errors propagate; other failures yield an analysis that says the sample is missing."""
        before = len(self._client.records)
        listings = []
        failure: str | None = None
        try:
            listings = fetch_jobs(self._client, step.query, refresh=refresh)
        except SerpApiError as exc:
            if exc.is_fatal:
                raise
            logger.warning("Market sample search failed: %s", exc.kind)
            failure = f"The market sample search failed: {exc.user_message}"
        finally:
            search_ids = [r.id for r in self._client.records[before:]]
            self._store.register_searches(self._client.records)
        analysis = analyze_market(
            opp, listings, self._store, role_query=step.query.q,
            location=step.query.location or self._default_location, resume=resume, search_ids=search_ids,
        )
        if failure:
            analysis.notes.append(failure)
        return analysis
