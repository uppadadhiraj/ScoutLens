"""Test doubles. ``ScriptedClient`` subclasses the real SerpApiClient and only replaces the HTTP call,
so parameter building, caching, record-keeping and error handling are the production code paths."""
from __future__ import annotations

import copy
from typing import Any

from config import Settings
from services.cache.search_cache import SearchCache
from services.llm.base import LLMProvider
from services.serpapi.client import SerpApiClient
from utils.errors import LLMUnavailableError, SerpApiError


class ScriptedClient(SerpApiClient):
    """Serves canned payloads by ``(engine, lower-cased q)``. A list value is a sequence of result pages;
    pagination tokens ("P1", "P2", ...) are injected automatically."""

    def __init__(
        self,
        settings: Settings,
        routes: dict[tuple[str, str], Any] | None = None,
        cache: SearchCache | None = None,
        errors: dict[tuple[str, str], SerpApiError] | None = None,
    ) -> None:
        super().__init__(settings, cache=cache, sleep=lambda _s: None)
        self.routes = {(e, q.lower()): v for (e, q), v in (routes or {}).items()}
        self.errors = {(e, q.lower()): v for (e, q), v in (errors or {}).items()}
        self.requested: list[tuple[str, str]] = []

    def _request(self, params: dict[str, Any], api_key: str) -> dict[str, Any]:
        key = (params["engine"], params["q"].lower())
        self.requested.append(key)
        if key in self.errors:
            raise self.errors[key]
        payload = self.routes.get(key, {})
        if not isinstance(payload, list):
            return copy.deepcopy(payload)
        token = params.get("next_page_token")
        index = int(token[1:]) if token else 0
        page = copy.deepcopy(payload[index]) if index < len(payload) else {}
        if index + 1 < len(payload):
            page["serpapi_pagination"] = {"next_page_token": f"P{index + 1}"}
        return page


class FakeLLM(LLMProvider):
    """Returns scripted replies in order; records every prompt it was given."""

    name = "fake"

    def __init__(self, replies: list[str] | None = None, available: bool = True) -> None:
        self.replies = list(replies or [])
        self.calls: list[tuple[str, str]] = []
        self._available = available

    def _chat(self, system: str, user: str, *, schema: dict | None, temperature: float) -> str:
        self.calls.append((system, user))
        if not self._available:
            raise LLMUnavailableError("fake LLM unavailable")
        if not self.replies:
            raise LLMUnavailableError("fake LLM has no scripted reply")
        return self.replies.pop(0)

    def check_available(self) -> tuple[bool, str]:
        return self._available, "fake"
