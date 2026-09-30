from datetime import datetime, timezone

import responses

from models.search import SearchQuery
from services.serpapi.client import SERPAPI_ENDPOINT
from services.serpapi.jobs import dedupe_jobs, fetch_jobs, next_page_token, parse_jobs
from services.serpapi.news import parse_news, parse_news_date
from services.serpapi.search import parse_google


def test_google_hits_drop_items_without_a_valid_url(google_payload):
    result = parse_google(google_payload, "s1")
    assert [h.position for h in result.hits] == [1, 2]
    assert all(h.url.startswith("https://") for h in result.hits)
    assert result.hits[1].date == "Sep 12, 2026"
    assert all(h.search_id == "s1" for h in result.hits)


def test_knowledge_graph_extracts_identity_and_scalar_attributes(google_payload):
    kg = parse_google(google_payload, "s1").knowledge_graph
    assert kg is not None
    assert kg.title == "Northwind Labs"
    assert kg.website == "https://www.northwindlabs.example"
    assert kg.source_name == "Wikipedia"
    assert kg.attributes == {"headquarters": "Bengaluru, India", "founded": "2016"}


def test_google_without_knowledge_graph(google_payload):
    del google_payload["knowledge_graph"]
    assert parse_google(google_payload, "s1").knowledge_graph is None


def test_parse_jobs_fields(jobs_payload):
    jobs = parse_jobs(jobs_payload, "s2")
    assert [j.title for j in jobs] == ["Backend Engineer", "Software Engineer Intern", "Backend Engineer"]
    first = jobs[0]
    assert first.via == "LinkedIn"  # "via " prefix stripped
    assert first.posted_at == "3 days ago" and first.schedule_type == "Full-time"
    assert first.salary == "₹12–18 LPA"
    assert first.qualifications == ["2+ years of Python", "Experience with SQL"]
    assert first.responsibilities == ["Design REST APIs", "Own deployments"]
    assert first.url == "https://www.northwindlabs.example/careers/backend"  # direct apply link wins
    assert jobs[1].work_from_home is True
    assert jobs[1].url.startswith("https://www.google.com/search")  # falls back to share link


def test_dedupe_by_job_id(jobs_payload):
    jobs = dedupe_jobs(parse_jobs(jobs_payload, "s2"))
    assert [j.job_id for j in jobs] == ["job-1", "job-2"]


def test_dedupe_without_job_id_uses_title_company_location():
    from models.search import JobListing

    a = JobListing(search_id="s1", title="Data Analyst", company_name="Acme", location="Pune")
    b = JobListing(search_id="s2", title="data analyst", company_name="ACME", location="pune")
    c = JobListing(search_id="s2", title="Data Analyst", company_name="Acme", location="Delhi")
    assert dedupe_jobs([a, b, c]) == [a, c]


def test_next_page_token(jobs_payload):
    assert next_page_token(jobs_payload) == "TOKEN_PAGE_2"
    assert next_page_token({}) is None


@responses.activate
def test_fetch_jobs_follows_pagination_and_dedupes(client, jobs_payload):
    page2 = {
        "jobs_results": [{"title": "Data Engineer", "company_name": "Northwind Labs", "job_id": "job-3"}],
    }
    responses.add(responses.GET, SERPAPI_ENDPOINT, json=jobs_payload)
    responses.add(responses.GET, SERPAPI_ENDPOINT, json=page2)

    jobs = fetch_jobs(client, SearchQuery(engine="google_jobs", q="Northwind Labs", pages=3))

    assert [j.job_id for j in jobs] == ["job-1", "job-2", "job-3"]
    assert len(responses.calls) == 2  # page 2 had no next token, so pagination stopped
    assert "next_page_token=TOKEN_PAGE_2" in responses.calls[1].request.url
    assert [r.id for r in client.records] == ["s1", "s2"]


@responses.activate
def test_fetch_jobs_keeps_page_one_when_page_two_fails(client, jobs_payload):
    responses.add(responses.GET, SERPAPI_ENDPOINT, json=jobs_payload)
    responses.add(responses.GET, SERPAPI_ENDPOINT, json={"error": "Too many requests"}, status=429)

    jobs = fetch_jobs(client, SearchQuery(engine="google_jobs", q="x", pages=2))

    assert [j.job_id for j in jobs] == ["job-1", "job-2"]
    assert client.records[1].error is not None


def test_parse_news_flattens_stories_and_drops_unlinked(news_payload):
    items = parse_news(news_payload, "s3")
    assert [i.title for i in items] == [
        "Northwind Labs raises Series B to expand logistics platform",
        "Northwind Labs launches route planner",
        "Analysts on Northwind's new product",
    ]
    assert items[0].publisher == "Example Business Daily"
    assert items[2].publisher == "Analysis Weekly"  # source given as a plain string


def test_news_dates_are_parsed_when_absolute_and_kept_as_text_when_relative(news_payload):
    items = parse_news(news_payload, "s3")
    assert items[0].published_at == datetime(2026, 9, 29, 8, 7, tzinfo=timezone.utc)
    assert items[1].published_at is None and items[1].date_text == "3 days ago"


def test_parse_news_date_edge_cases():
    assert parse_news_date(None) is None
    assert parse_news_date("yesterday") is None
    assert parse_news_date("2026-09-01") == datetime(2026, 9, 1, tzinfo=timezone.utc)


def test_parsers_tolerate_empty_and_malformed_payloads():
    assert parse_jobs({}, "s") == []
    assert parse_news({"news_results": "not a list"}, "s") == []
    assert parse_google({"organic_results": [None, 5, "x"]}, "s").hits == []
