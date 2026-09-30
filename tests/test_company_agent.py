import copy

import pytest

from agents.company_agent import CompanyAgent, _domain_matches_company
from agents.planner import InvestigationPlanner
from models.opportunity import OpportunityProfile
from services.evidence.store import EvidenceStore
from tests.fakes import FakeLLM, ScriptedClient

OPP = OpportunityProfile(
    company_name="Northwind Labs",
    role_title="Software Engineer Intern",
    application_url="https://www.northwindlabs.example/careers/swe-intern",
)
IDENTITY = ("google", "Northwind Labs")
FUNDING = ("google", "Northwind Labs funding")


@pytest.fixture
def plan(settings):
    return InvestigationPlanner(settings).initial_plan(OPP)


def _agent(settings, routes, llm=None):
    client = ScriptedClient(settings, routes)
    store = EvidenceStore(OPP.company_name)
    return CompanyAgent(client, store, llm), client, store


def test_identity_from_knowledge_panel(settings, plan, google_payload):
    agent, client, store = _agent(settings, {IDENTITY: google_payload})
    result = agent.investigate(OPP, plan.step("identity"))
    p = result.profile

    assert p.website == "https://www.northwindlabs.example"
    assert p.website_basis == "Listed as the website in Google's knowledge panel"
    assert p.identity_confirmed and result.has_knowledge_graph
    assert p.industry == "Software company"
    assert p.overview.startswith("Northwind Labs builds logistics") and p.overview_source == "knowledge_graph"
    assert result.kg_founded_year == 2016
    assert p.attributes["headquarters"].text == "Bengaluru, India"
    assert p.gaps == []

    # every populated field points at real evidence, and the website evidence is Tier 1 (official domain)
    for eid in [*p.website_evidence_ids, *p.overview_evidence_ids, *p.attributes["founded"].evidence_ids]:
        assert store.get(eid) is not None
    assert store.get(p.website_evidence_ids[0]).tier == 1
    # the overview's own source is Wikipedia, which is community-edited -> Tier 3, and the UI can show that
    assert store.get(p.overview_evidence_ids[0]).tier == 3


def test_product_facts_are_verbatim_official_site_excerpts(settings, plan, google_payload):
    agent, _, store = _agent(settings, {IDENTITY: google_payload})
    p = agent.investigate(OPP, plan.step("identity")).profile
    assert [f.text for f in p.products] == [
        "Northwind Labs helps retailers plan routes and inventory.", "Join our engineering team."
    ]
    assert all(store.get(f.evidence_ids[0]).tier == 1 for f in p.products)


def test_without_knowledge_panel_website_comes_from_the_listing_domain(settings, plan, google_payload):
    del google_payload["knowledge_graph"]
    agent, _, _ = _agent(settings, {IDENTITY: google_payload})
    p = agent.investigate(OPP, plan.step("identity")).profile
    assert p.website == "https://www.northwindlabs.example/"
    assert p.website_basis == "The listing itself is hosted on this domain"
    assert p.overview_source == "search_snippet"
    assert p.identity_confirmed


def test_website_inferred_from_search_results_is_labelled_as_inferred(settings, plan, google_payload):
    del google_payload["knowledge_graph"]
    opp = OPP.model_copy(update={"application_url": "https://www.linkedin.com/jobs/view/1"})
    agent, _, _ = _agent(settings, {IDENTITY: google_payload})
    p = agent.investigate(opp, plan.step("identity")).profile
    assert p.website == "https://www.northwindlabs.example/"
    assert p.website_basis.startswith("Inferred")


def test_no_evidence_means_explicit_gaps_not_invention(settings, plan):
    agent, _, store = _agent(settings, {IDENTITY: {}})
    opp = OPP.model_copy(update={"application_url": "https://www.linkedin.com/jobs/view/1"})
    p = agent.investigate(opp, plan.step("identity")).profile
    assert p.website is None and p.overview is None and not p.identity_confirmed
    assert any("official website" in g and g.startswith("Insufficient evidence") for g in p.gaps)
    assert any("overview" in g for g in p.gaps)
    assert len(store) == 0


def test_unrelated_results_are_flagged_as_ambiguous_and_not_used(settings, plan):
    noise = {"organic_results": [
        {"position": i, "title": f"Unrelated page {i}", "link": f"https://other{i}.example/", "snippet": "Nothing relevant"}
        for i in range(1, 6)
    ]}
    agent, _, store = _agent(settings, {IDENTITY: noise})
    opp = OPP.model_copy(update={"application_url": "https://www.linkedin.com/jobs/view/1"})
    result = agent.investigate(opp, plan.step("identity"))
    assert result.profile.ambiguous and result.relevant_hits == []
    assert any("did not clearly identify" in g for g in result.profile.gaps)
    assert len(store) == 0


def test_funding_facts_are_verbatim_and_cited(settings, plan, google_payload):
    funding = {"organic_results": [
        {"position": 1, "title": "Northwind Labs raises $5M seed round", "link": "https://startupnews.example/northwind",
         "snippet": "Northwind Labs raised a $5M seed round led by Acme Ventures."},
        {"position": 2, "title": "Northwind Labs careers", "link": "https://www.northwindlabs.example/careers", "snippet": "Join us."},
        {"position": 3, "title": "Some other company raises funding", "link": "https://x.example/y", "snippet": "Other Corp raised $9M."},
    ]}
    agent, _, store = _agent(settings, {IDENTITY: google_payload, FUNDING: funding})
    result = agent.investigate(OPP, plan.step("identity"))
    step = InvestigationPlanner(settings).follow_ups(OPP, __import__("models.plan", fromlist=["PlanState"]).PlanState(), set())[0]
    funding_step = next(s for s in step if s.key == "funding")
    agent.add_funding(result, funding_step)

    facts = result.profile.funding
    assert [f.text for f in facts] == ["Northwind Labs raised a $5M seed round led by Acme Ventures."]
    assert store.get(facts[0].evidence_ids[0]).source_url == "https://startupnews.example/northwind"


def test_missing_funding_is_reported_as_not_found_rather_than_none_exists(settings, plan, google_payload):
    agent, _, _ = _agent(settings, {IDENTITY: google_payload, FUNDING: {}})
    result = agent.investigate(OPP, plan.step("identity"))
    from models.plan import PlanState

    steps, _ = InvestigationPlanner(settings).follow_ups(OPP, PlanState(), set())
    agent.add_funding(result, next(s for s in steps if s.key == "funding"))
    assert result.profile.funding == []
    assert "No funding information was found in the retrieved results." in result.profile.gaps


def test_funding_from_news_is_added_with_news_citations(settings, plan, google_payload, news_payload):
    from services.serpapi.news import parse_news

    agent, _, store = _agent(settings, {IDENTITY: google_payload})
    result = agent.investigate(OPP, plan.step("identity"))
    result.profile.gaps.append("No funding information was found in the retrieved results.")
    news = parse_news(news_payload, "s9")
    agent.add_funding_from_news(result.profile, news)
    assert len(result.profile.funding) == 1
    assert "Series B" in store.get(result.profile.funding[0].evidence_ids[0]).source_title
    assert not any(g.startswith("No funding") for g in result.profile.gaps)


# ------------------------------------------------------------------ LLM overview

def _no_kg(payload):
    payload = copy.deepcopy(payload)
    del payload["knowledge_graph"]
    return payload


def test_llm_overview_keeps_only_cited_sentences(settings, plan, google_payload):
    # E1 is the website claim for the same hit; the description excerpt the LLM is shown is E2
    llm = FakeLLM(["Northwind Labs helps retailers plan routes and inventory [E2]. It is the best company ever. It has a Mars office [E77]."])
    agent, _, store = _agent(settings, {IDENTITY: _no_kg(google_payload)}, llm)
    p = agent.investigate(OPP, plan.step("identity")).profile
    assert p.overview == "Northwind Labs helps retailers plan routes and inventory [E2]."
    assert p.overview_source == "llm" and p.overview_evidence_ids == ["E2"]
    assert "Excerpts" in llm.calls[0][1] and "[E2]" in llm.calls[0][1]


def test_llm_overview_falls_back_when_text_is_not_neutral(settings, plan, google_payload):
    llm = FakeLLM(["Northwind Labs is a definitely good company for engineers [E2]."])
    agent, _, _ = _agent(settings, {IDENTITY: _no_kg(google_payload)}, llm)
    p = agent.investigate(OPP, plan.step("identity")).profile
    assert p.overview_source == "search_snippet"


def test_llm_overview_falls_back_when_llm_is_down(settings, plan, google_payload):
    agent, _, _ = _agent(settings, {IDENTITY: _no_kg(google_payload)}, FakeLLM(available=False))
    p = agent.investigate(OPP, plan.step("identity")).profile
    assert p.overview_source == "search_snippet" and p.overview


def test_llm_output_with_no_valid_citations_is_discarded(settings, plan, google_payload):
    agent, _, _ = _agent(settings, {IDENTITY: _no_kg(google_payload)}, FakeLLM(["A confident but uncited claim."]))
    assert agent.investigate(OPP, plan.step("identity")).profile.overview_source == "search_snippet"


@pytest.mark.parametrize(
    "url, company, expected",
    [
        ("https://www.northwindlabs.com/", "Northwind Labs", True),
        ("https://northwind-labs.com/", "Northwind Labs", True),  # hyphens are ignored
        ("https://northwind.com/", "Northwind Labs", True),  # one trailing generic word ("Labs") may be dropped
        ("https://northwindlabs.co.in/careers", "Northwind Labs Pvt Ltd", True),
        ("https://careers.northwindlabs.com/x", "Northwind Labs", True),  # subdomains do not matter
        # lookalikes must never become "official" (that would earn them Tier 1)
        ("https://northwindlabs-review.com/", "Northwind Labs", False),
        ("https://northwindlabsjobs.com/", "Northwind Labs", False),
        ("https://northwind.com/", "Northwind Robotics", False),
        ("https://acme.com/", "Northwind Labs", False),
    ],
)
def test_domain_matches_company(url, company, expected):
    assert _domain_matches_company(url, company) is expected
