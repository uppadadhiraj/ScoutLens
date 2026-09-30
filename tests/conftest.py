from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import Settings
from services.cache.search_cache import SearchCache
from services.serpapi.client import SerpApiClient

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def google_payload() -> dict:
    return load_fixture("serpapi_google")


@pytest.fixture
def jobs_payload() -> dict:
    return load_fixture("serpapi_jobs")


@pytest.fixture
def news_payload() -> dict:
    return load_fixture("serpapi_news")


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings with a fake key and an isolated DB; never reads the developer's real .env."""
    return Settings(_env_file=None, serpapi_key="test-key-123456", db_path=tmp_path / "test.db")


@pytest.fixture
def cache(settings) -> SearchCache:
    return SearchCache(settings.db_path)


@pytest.fixture
def client(settings, cache) -> SerpApiClient:
    return SerpApiClient(settings, cache=cache, sleep=lambda _s: None)
