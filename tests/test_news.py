import json
from datetime import datetime, timezone

import pytest

from agents.news_agent import NewsAgent
from agents.planner import InvestigationPlanner
from models.opportunity import OpportunityProfile
from models.search import NewsItem
from services.analysis.news import analyze_news, approx_age_days, classify_topic, role_terms
from services.evidence.store import EvidenceStore
from services.serpapi.news import parse_news
from tests.fakes import FakeLLM, ScriptedClient
from utils.errors import SerpApiError
from utils.language_guard import is_neutral
from utils.text import brand_alias, mentions_company

CO = "Northwind Labs"
NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
OPP = OpportunityProfile(company_name=CO, role_title="Software Engineer Intern", required_skills=["Python", "Go", "R"])


def news(title, snippet="", publisher="Reuters", date_text=None, published=None, url=None):
    return NewsItem(
        search_id="s3", title=title, url=url or f"https://news.example.com/{abs(hash(title))}", publisher=publisher,
        snippet=snippet, date_text=date_text, published_at=published,
    )


def run(items, opp=OPP, **kw):
    store = EvidenceStore(CO)
    signal, relevant = analyze_news(opp, items, store, now=NOW, **kw)
    return signal, relevant, store


# -------------------------------------------------------------------- helpers

@pytest.mark.parametrize(
    "text, days",
    [("3 days ago", 3), ("1 hour ago", 0), ("2 weeks ago", 14), ("1 month ago", 30), ("2 years ago", 730),
     ("09/29/2026, 08:07 AM, +0000 UTC", None), (None, None), ("yesterday", None)],
)
def test_approx_age_days(text, days):
    assert approx_age_days(text) == days


@pytest.mark.parametrize(
    "text, topic",
    [
        ("Northwind Labs raises $5M Series A", "Funding & investment"),
        ("Company announces layoffs across teams", "Workforce changes"),
        ("Northwind sued over patent dispute", "Legal & regulatory"),
        ("Northwind launches route planner", "Product & launches"),
        ("Northwind partners with Acme", "Partnerships"),
        ("Northwind opens new office in Pune", "Expansion & hiring"),
        ("Northwind appoints new CTO", "Leadership"),
        ("Northwind at the trade show", "General coverage"),
    ],
)
def test_classify_topic(text, topic):
    assert classify_topic(text) == topic


def test_brand_alias_and_mentions():
    assert brand_alias("Northwind Labs") == "northwind" and brand_alias("Go Labs") is None and brand_alias("Acme") is None
    assert mentions_company("Analysts on Northwind's new product", CO)  # short brand form counts
    assert mentions_company("NORTHWIND LABS raises money", CO)
    assert not mentions_company("Northwindsor Castle opens", CO)  # whole-token match only
    assert not mentions_company("Something unrelated", CO)


def test_role_terms_exclude_short_skill_names():
    terms = role_terms(OPP)
    assert {"software", "engineer", "python"} <= terms
    assert "go" not in terms and "r" not in terms


# --------------------------------------------------------------------- analysis

def test_fixture_news_is_filtered_ordered_and_cited(news_payload):
    signal, relevant, store = run(parse_news(news_payload, "s3"))
    assert [n.title for n in signal.items] == [
        "Northwind Labs raises Series B to expand logistics platform",  # 1 day old
        "Northwind Labs launches route planner",  # "3 days ago"
        "Analysts on Northwind's new product",  # 10 days old
    ]
    first = signal.items[0]
    assert (first.topic, first.publisher, first.tier) == ("Funding & investment", "Example Business Daily", 3)
    assert first.summary.startswith("The Bengaluru-based company said")
    assert store.get(first.evidence_id).source_url == first.url
    assert len(relevant) == 3 and signal.total_found == 3


def test_dates_are_never_invented():
    signal, _, _ = run([news("Northwind Labs launches app", date_text="3 days ago")])
    item = signal.items[0]
    assert item.published_at is None and item.date_text == "3 days ago"


def test_role_related_news_ranks_first_even_if_older():
    items = [
        news("Northwind Labs launches app", date_text="1 day ago"),
        news("Northwind Labs hiring software engineer interns in Pune", date_text="20 days ago"),
    ]
    signal, _, _ = run(items)
    assert signal.items[0].title.startswith("Northwind Labs hiring software engineer")
    assert signal.items[0].relevance == "high"
    assert signal.items[0].relevance_reason.startswith("Mentions '") and "this role's title or requirements" in signal.items[0].relevance_reason
    assert signal.items[1].relevance == "medium"


def test_general_coverage_is_low_relevance():
    signal, _, _ = run([news("Northwind Labs at the trade show")])
    assert signal.items[0].relevance == "low" and signal.items[0].topic == "General coverage"


def test_unrelated_and_old_results_are_excluded_and_counted():
    items = [
        news("Someone else raises money"),
        news("Northwind Labs old story", published=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        news("Northwind Labs ancient story", date_text="3 years ago"),
        news("Northwind Labs launches app", published=datetime(2026, 9, 1, tzinfo=timezone.utc)),
    ]
    signal, _, _ = run(items)
    assert [n.title for n in signal.items] == ["Northwind Labs launches app"]
    assert (signal.excluded_irrelevant, signal.excluded_old) == (1, 2)
    assert any("did not mention the company" in n for n in signal.notes)
    assert any("older than 365 days" in n for n in signal.notes)


def test_unknown_dates_are_kept_but_ranked_after_dated_items_of_equal_relevance():
    signal, _, _ = run([news("Northwind Labs launches A"), news("Northwind Labs launches B", date_text="20 days ago")])
    assert [n.title[-1] for n in signal.items] == ["B", "A"]


def test_duplicate_urls_are_collapsed_and_max_items_caps_the_list():
    same = news("Northwind Labs launches A", url="https://n.example/a")
    signal, _, _ = run([same, same.model_copy(update={"url": "https://n.example/a?utm=1"})])
    assert len(signal.items) == 1
    many = [news(f"Northwind Labs launches {i}", url=f"https://n.example/{i}") for i in range(10)]
    assert len(run(many, max_items=4)[0].items) == 4


def test_no_news_is_reported_as_insufficient_evidence():
    signal, relevant, store = run([])
    assert signal.items == [] and relevant == [] and len(store) == 0
    assert any(n.startswith("Insufficient evidence") for n in signal.notes)


def test_all_generated_wording_is_neutral():
    items = [news(t) for t in ("Northwind Labs announces layoffs", "Northwind Labs sued in court", "Northwind Labs raises Series A",
                                "Northwind Labs partners with Acme", "Northwind Labs at expo")]
    signal, _, _ = run(items)
    for n in signal.items:
        assert is_neutral(f"{n.topic} {n.relevance_reason}"), n


# ------------------------------------------------------------------------ agent

@pytest.fixture
def steps(settings):
    plan = InvestigationPlanner(settings).initial_plan(OPP)
    return [plan.step("news")]


def _agent(settings, routes, llm=None, errors=None):
    client = ScriptedClient(settings, routes, errors=errors)
    return NewsAgent(client, EvidenceStore(CO), llm), client


def test_agent_runs_the_news_search_and_records_the_search_id(settings, steps, news_payload):
    agent, client = _agent(settings, {("google_news", CO): news_payload})
    signal, relevant = agent.investigate(OPP, steps, max_items=8)
    assert len(signal.items) == 3 and signal.search_ids == ["s1"] and client.records[0].engine == "google_news"


def test_agent_fatal_error_propagates_but_others_become_notes(settings, steps):
    agent, _ = _agent(settings, {}, errors={("google_news", CO.lower()): SerpApiError(kind="timeout")})
    signal, _ = agent.investigate(OPP, steps)
    assert signal.items == [] and any("failed" in n for n in signal.notes)
    agent, _ = _agent(settings, {}, errors={("google_news", CO.lower()): SerpApiError(kind="rate_limited")})
    with pytest.raises(SerpApiError):
        agent.investigate(OPP, steps)


def _judge(*rows):
    return json.dumps({"items": [{"index": i, "relevance": r, "why": w} for i, r, w in rows]})


def test_llm_refinement_updates_relevance_and_reorders(settings, steps, news_payload):
    llm = FakeLLM([_judge((0, "low", "General funding news."), (2, "high", "Discusses the new product this role may support."))])
    agent, _ = _agent(settings, {("google_news", CO): news_payload}, llm)
    signal, _ = agent.investigate(OPP, steps)
    assert signal.items[0].title == "Analysts on Northwind's new product" and signal.items[0].relevance == "high"
    assert signal.items[0].relevance_reason == "Discusses the new product this role may support."
    assert signal.items[-1].relevance == "low"
    assert "Northwind Labs raises Series B" in llm.calls[0][1]


def test_llm_refinement_ignores_bad_indexes_and_non_neutral_reasons(settings, steps, news_payload):
    llm = FakeLLM([_judge((0, "high", "This is definitely a great company."), (99, "high", "Out of range."), (1, "low", "A neutral reason."))])
    agent, _ = _agent(settings, {("google_news", CO): news_payload}, llm)
    signal, _ = agent.investigate(OPP, steps)
    by_title = {n.title: n for n in signal.items}
    assert by_title["Northwind Labs raises Series B to expand logistics platform"].relevance == "medium"  # untouched
    assert by_title["Northwind Labs launches route planner"].relevance == "low"


def test_llm_unavailable_leaves_deterministic_results(settings, steps, news_payload):
    agent, _ = _agent(settings, {("google_news", CO): news_payload}, FakeLLM(available=False))
    signal, _ = agent.investigate(OPP, steps)
    assert len(signal.items) == 3 and signal.items[0].relevance == "medium"
