from urllib.parse import parse_qs, urlparse

import pytest
import requests
import responses

from config import Settings
from models.search import SearchQuery
from services.serpapi.client import SERPAPI_ENDPOINT, SerpApiClient
from utils.errors import SerpApiError

Q = SearchQuery(engine="google", q="Northwind Labs", purpose="identify company")


def _sent_params(call) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlparse(call.request.url).query).items()}


@responses.activate
def test_success_sends_expected_params_and_records_search(client, google_payload):
    responses.add(responses.GET, SERPAPI_ENDPOINT, json=google_payload)
    resp = client.search(Q)

    sent = _sent_params(responses.calls[0])
    assert sent["engine"] == "google" and sent["q"] == "Northwind Labs"
    assert sent["gl"] == "in" and sent["hl"] == "en" and sent["location"] == "India"
    assert sent["api_key"] == "test-key-123456"

    assert resp.search_id == "s1"
    assert resp.record.result_count == 4
    assert resp.record.serpapi_search_id == "abc123"
    assert not resp.record.from_cache
    assert "test-key-123456" not in resp.record.model_dump_json()


@responses.activate
def test_news_engine_does_not_send_location(client, news_payload):
    responses.add(responses.GET, SERPAPI_ENDPOINT, json=news_payload)
    client.search(SearchQuery(engine="google_news", q="Northwind Labs"))
    assert "location" not in _sent_params(responses.calls[0])


@responses.activate
def test_second_identical_search_is_served_from_cache(client, google_payload):
    responses.add(responses.GET, SERPAPI_ENDPOINT, json=google_payload)
    first = client.search(Q)
    second = client.search(Q)

    assert len(responses.calls) == 1
    assert second.record.from_cache and not first.record.from_cache
    assert second.data == first.data
    assert [r.id for r in client.records] == ["s1", "s2"]


@responses.activate
def test_refresh_bypasses_cache_and_updates_it(client, google_payload):
    responses.add(responses.GET, SERPAPI_ENDPOINT, json=google_payload)
    client.search(Q)
    client.search(Q, refresh=True)
    assert len(responses.calls) == 2
    assert client.search(Q).record.from_cache  # refreshed entry is reused afterwards


@responses.activate
def test_zero_ttl_disables_cache(tmp_path, cache, google_payload):
    settings = Settings(_env_file=None, serpapi_key="test-key-123456", cache_ttl_hours=0, db_path=tmp_path / "t.db")
    client = SerpApiClient(settings, cache=cache, sleep=lambda _s: None)
    responses.add(responses.GET, SERPAPI_ENDPOINT, json=google_payload)
    client.search(Q)
    client.search(Q)
    assert len(responses.calls) == 2


@responses.activate
def test_missing_key_raises_without_any_http_call(tmp_path):
    settings = Settings(_env_file=None, db_path=tmp_path / "t.db")
    client = SerpApiClient(settings, sleep=lambda _s: None)
    with pytest.raises(SerpApiError) as info:
        client.search(Q)
    assert info.value.kind == "no_key" and info.value.is_fatal
    assert "SERPAPI_KEY" in info.value.user_message
    assert len(responses.calls) == 0
    assert client.records[0].error  # failed searches are still audited


@responses.activate
def test_invalid_key_is_fatal_and_message_is_safe(client):
    responses.add(responses.GET, SERPAPI_ENDPOINT, json={"error": "Invalid API key. Your API key should be here: https://serpapi.com/manage-api-key"}, status=401)
    with pytest.raises(SerpApiError) as info:
        client.search(Q)
    assert info.value.kind == "invalid_key" and info.value.is_fatal
    assert "test-key-123456" not in info.value.user_message


@responses.activate
def test_rate_limit_http_429(client):
    responses.add(responses.GET, SERPAPI_ENDPOINT, json={"error": "Too many requests"}, status=429)
    with pytest.raises(SerpApiError) as info:
        client.search(Q)
    assert info.value.kind == "rate_limited"


@responses.activate
def test_out_of_searches_message_maps_to_rate_limited(client):
    responses.add(responses.GET, SERPAPI_ENDPOINT, json={"error": "Your account has run out of searches."}, status=200)
    with pytest.raises(SerpApiError) as info:
        client.search(Q)
    assert info.value.kind == "rate_limited"


@responses.activate
def test_no_results_message_is_an_empty_result_not_an_error(client):
    responses.add(responses.GET, SERPAPI_ENDPOINT, json={"error": "Google hasn't returned any results for this query."})
    resp = client.search(Q)
    assert resp.record.result_count == 0 and resp.record.error is None
    assert "error" not in resp.data


@responses.activate
def test_server_error_is_retried_once_then_succeeds(client, google_payload):
    responses.add(responses.GET, SERPAPI_ENDPOINT, status=503)
    responses.add(responses.GET, SERPAPI_ENDPOINT, json=google_payload)
    assert client.search(Q).record.result_count == 4
    assert len(responses.calls) == 2


@responses.activate
def test_timeout_retries_then_raises_user_safe_error(client):
    responses.add(responses.GET, SERPAPI_ENDPOINT, body=requests.Timeout("timed out"))
    with pytest.raises(SerpApiError) as info:
        client.search(Q)
    assert info.value.kind == "timeout"
    assert len(responses.calls) == 2


@responses.activate
def test_connection_error_detail_never_contains_the_api_key(client):
    leaky = requests.ConnectionError(
        "HTTPSConnectionPool(host='serpapi.com', port=443): Max retries exceeded with url: "
        "/search.json?engine=google&api_key=test-key-123456&q=x"
    )
    responses.add(responses.GET, SERPAPI_ENDPOINT, body=leaky)
    with pytest.raises(SerpApiError) as info:
        client.search(Q)
    assert info.value.kind == "network"
    assert "test-key-123456" not in str(info.value)
    assert "test-key-123456" not in (info.value.detail or "")
    assert "test-key-123456" not in info.value.user_message


@responses.activate
def test_non_json_response_is_bad_response(client):
    responses.add(responses.GET, SERPAPI_ENDPOINT, body="<html>oops</html>", status=200)
    with pytest.raises(SerpApiError) as info:
        client.search(Q)
    assert info.value.kind == "bad_response"
