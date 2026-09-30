"""Hiring Agent: runs the planned Google Jobs searches and turns the results into a HiringSignal."""
from __future__ import annotations

import logging

from models.opportunity import OpportunityProfile
from models.plan import PlannedSearch
from models.report import HiringSignal
from models.search import JobListing
from services.analysis.hiring import analyze_hiring
from services.evidence.store import EvidenceStore
from services.serpapi.client import SerpApiClient
from services.serpapi.jobs import fetch_jobs
from utils.errors import SerpApiError

logger = logging.getLogger(__name__)


class HiringAgent:
    def __init__(self, client: SerpApiClient, store: EvidenceStore) -> None:
        self._client = client
        self._store = store
        self.errors: list[str] = []  # non-fatal failures, surfaced as report warnings
        self._listings: list[JobListing] = []  # accumulated across rounds (initial searches + fallbacks)
        self._search_ids: list[str] = []

    def collect(self, steps: list[PlannedSearch], *, refresh: bool = False) -> tuple[list[JobListing], list[str]]:
        """Run each job search; returns ``(listings, ids of the searches that were executed)``.

        Fatal SerpApi errors (bad key, quota) propagate so the investigator can stop cleanly; other
        failures are recorded in ``errors`` and the remaining searches still run.
        """
        listings: list[JobListing] = []
        search_ids: list[str] = []
        for step in steps:
            before = len(self._client.records)
            try:
                listings += fetch_jobs(self._client, step.query, refresh=refresh)
            except SerpApiError as exc:
                if exc.is_fatal:
                    raise
                logger.warning("Job search '%s' failed: %s", step.key, exc.kind)
                self.errors.append(f"Job search '{step.query.q}' failed: {exc.user_message}")
            finally:
                search_ids += [r.id for r in self._client.records[before:]]
                self._store.register_searches(self._client.records)
        return listings, search_ids

    def investigate(
        self, opp: OpportunityProfile, steps: list[PlannedSearch], *, refresh: bool = False
    ) -> HiringSignal:
        """Run ``steps`` and analyse everything collected so far (safe to call again with follow-up steps)."""
        listings, search_ids = self.collect(steps, refresh=refresh)
        self._listings += listings
        self._search_ids += search_ids
        signal = analyze_hiring(opp, self._listings, self._store, search_ids=self._search_ids)
        signal.notes.extend(self.errors)
        return signal
