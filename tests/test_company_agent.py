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
IDENTITY = ("google", "Northwind Labs company")
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
    # "Join our engineering team." (26 chars) is too thin to describe the business, so it is not offered as an excerpt
    assert [f.text for f in p.products] == ["Northwind Labs helps retailers plan routes and inventory."]
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


# ---- lessons from real Google responses ----------------------------------------------------------------

from agents.company_agent import is_descriptive, is_funding_fact  # noqa: E402
from models.search import SearchHit  # noqa: E402
from services.evidence.citations import grounding_score, validate_prose  # noqa: E402


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Spotify has raised a total of $2.1B of funding over 22 rounds. Their latest funding round was in 2023.", True),
        ("Northwind Labs raised a $5M seed round led by Acme Ventures.", True),
        ("Northwind Labs raises Series B to expand its platform", True),
        ("Northwind is acquired by Globex", True),
        ("Spotify's 2026 Investor Day Recap: Raising Ambition for the Next Era of Media", False),
        ("Spotify's founder funding military AI, of its founder's investment in military tech", False),
        ("Spotify has raised a single venture fund. Spotify is funded by 53 investors.", False),
        ("Funding your next playlist", False),
    ],
)
def test_funding_facts_must_name_an_amount_or_a_round(text, expected):
    assert is_funding_fact(text) is expected


def _hit(url, snippet, title="T"):
    return SearchHit(search_id="s1", title=title, url=url, snippet=snippet)


@pytest.mark.parametrize(
    "url, snippet, expected",
    [
        ("https://www.acme.example/about", "Acme builds logistics optimisation software for retailers worldwide.", True),
        ("https://www.acme.example/us/login/?continue=/x", "Acme builds logistics optimisation software for retailers.", False),
        ("https://www.acme.example/about", "No information is available for this page.", False),
        ("https://www.acme.example/about", "Join our team.", False),
        ("https://open.acme.example/album/7mtRex3XcxOcJJ9", "202320 songs, 1 hr 12 min. Therefore I Am · Billie Eilish", False),
        ("https://www.youtube.com/shorts/abc", "Will Acme be the first media company to hit 1 billion subscribers?", False),
    ],
)
def test_only_descriptive_pages_can_describe_the_business(url, snippet, expected):
    assert is_descriptive(_hit(url, snippet)) is expected


def test_the_employers_about_page_outranks_other_official_pages(settings, plan):
    agent, _, store = _agent(settings, {})
    store.register_official_domain("https://www.northwindlabs.example/")
    long = "Northwind Labs builds logistics optimisation software for retailers worldwide."
    ranked = agent._descriptive_hits([
        _hit("https://other.example/blog", long),
        _hit("https://www.northwindlabs.example/pricing", long),
        _hit("https://www.northwindlabs.example/about", long),
    ])
    assert [h.url for h in ranked] == [
        "https://www.northwindlabs.example/about", "https://www.northwindlabs.example/pricing", "https://other.example/blog",
    ]


def test_funding_search_keeps_only_concrete_facts(settings, plan, google_payload):
    from models.plan import PlanState

    noisy = {"organic_results": [
        {"position": 1, "title": "Northwind Labs Investor Day Recap", "link": "https://a.example/1", "snippet": "Northwind Labs on raising ambition for the next era."},
        {"position": 2, "title": "Northwind Labs raised $9M", "link": "https://b.example/2", "snippet": "Northwind Labs raised $9M in a Series A round."},
    ]}
    agent, _, _ = _agent(settings, {IDENTITY: google_payload, FUNDING: noisy})
    result = agent.investigate(OPP, plan.step("identity"))
    steps, _ = InvestigationPlanner(settings).follow_ups(OPP, PlanState(), set())
    agent.add_funding(result, next(s for s in steps if s.key == "funding"))
    assert [f.text for f in result.profile.funding] == ["Northwind Labs raised $9M in a Series A round."]


# ---- LLM prose must be supported by its citations AND be about the company -------------------------------

def test_grounding_score_counts_content_words_found_in_the_cited_excerpt():
    excerpt = "Northwind Labs helps retailers plan routes and inventory with optimisation software."
    assert grounding_score("Northwind Labs helps retailers plan routes [E1].", excerpt, {"northwind", "labs"}) == 1.0
    assert grounding_score("The firm employs astronauts and submarine pilots [E1].", excerpt) < 0.3
    assert grounding_score("Yes [E1].", excerpt) == 1.0  # no content words: nothing to contradict


def test_a_valid_citation_does_not_excuse_an_unsupported_sentence(settings, plan, google_payload):
    agent, _, store = _agent(settings, {IDENTITY: _no_kg(google_payload)})
    agent.investigate(OPP, plan.step("identity"))
    text = "Northwind Labs helps retailers plan routes and inventory [E2]. Northwind Labs owns a fleet of submarines in the Pacific [E2]."
    checked = validate_prose(text, store, min_grounding=0.5)
    assert checked.text == "Northwind Labs helps retailers plan routes and inventory [E2]." and checked.dropped == 1


def test_llm_overview_about_something_other_than_the_company_is_discarded(settings, plan, google_payload):
    # grounded in a real excerpt and correctly cited, yet about songs rather than the employer (seen with real data)
    llm = FakeLLM(["The service offers a selection of routes and inventory tools for retailers [E2]. The plan covers optimisation software [E2]."])
    agent, _, _ = _agent(settings, {IDENTITY: _no_kg(google_payload)}, llm)
    p = agent.investigate(OPP, plan.step("identity")).profile
    assert p.overview_source == "search_snippet"  # fell back to the verbatim excerpt


def test_llm_overview_keeps_sentences_that_start_with_it(settings, plan, google_payload):
    llm = FakeLLM(["Northwind Labs helps retailers plan routes [E2]. It helps with inventory planning too [E2]."])
    agent, _, _ = _agent(settings, {IDENTITY: _no_kg(google_payload)}, llm)
    p = agent.investigate(OPP, plan.step("identity")).profile
    assert p.overview_source == "llm" and p.overview.count("[E2]") == 2


@pytest.mark.parametrize(
    "text, expected",
    [
        ("What Investors Should Know About Spotify Co-CEO Soderstrom Selling $10.6M Stock", False),  # seen with real data
        ("Spotify shares rise after earnings; investors cheer $2B buyback", False),
        ("Acme raised $30M in a funding round led by Demo Ventures", True),
        ("Acme secures $12M investment from Demo Ventures", True),
        ("Acme is acquired by Globex", True),
        ("Acme raises Series C", True),
    ],
)
def test_stock_sales_and_earnings_are_not_funding_facts(text, expected):
    assert is_funding_fact(text) is expected


def test_funding_facts_must_be_about_the_company_not_a_third_party_that_mentions_it():
    blissclub = "Blissclub founder's stake slips below 40% after Series B; Meesho's Vidit Aatrey holds 6.94%"
    assert is_funding_fact(blissclub) is True and is_funding_fact(blissclub, "Meesho") is False  # Series B is Blissclub's
    assert is_funding_fact("In September 2021, Meesho raised $570 million at a valuation of $4.9 billion", "Meesho")
    assert is_funding_fact("How Meesho's pitch deck secured $3.5B funding", "Meesho")
    assert is_funding_fact("Northwind Labs is acquired by Globex", "Northwind Labs")
    assert not is_funding_fact("Some other company raised $5M", "Northwind Labs")  # company not mentioned at all


def test_a_follow_up_search_that_finds_the_company_also_fills_the_profile(settings, plan, google_payload):
    """Real case: a generic-word first search ('Meesho company' returns dictionary entries) leaves the profile empty;
    the follow-up 'official website' search finds the company and its results must not be thrown away."""
    from models.plan import PlanState

    noise = {"organic_results": [
        {"position": i, "title": f"Company definition {i}", "link": f"https://dictionary{i}.example/company",
         "snippet": "A company is a legal entity representing an association of people."}
        for i in range(1, 6)
    ]}
    official = _no_kg(google_payload)
    agent, client, store = _agent(settings, {IDENTITY: noise, ("google", "Northwind Labs official website"): official})
    from_linkedin = OPP.model_copy(update={"application_url": "https://www.linkedin.com/jobs/view/1"})
    result = agent.investigate(from_linkedin, plan.step("identity"))
    assert result.profile.ambiguous and result.profile.website is None and result.profile.overview is None

    step = next(s for s in InvestigationPlanner(settings).follow_ups(OPP, PlanState(), set())[0] if s.key == "official_site")
    agent.retry_website(result, from_linkedin, step)
    p = result.profile
    assert p.website == "https://www.northwindlabs.example/" and p.identity_confirmed and not p.ambiguous
    assert p.products and p.overview_source == "search_snippet" and p.overview
    assert p.gaps == []


# ---- real case: an official storefront's ad copy vs the company's own profile page --------------------------

PROMO = "Grab Exciting Deals: Unbeatable Prices on Must-Haves with free shipping and easy returns on every order."
PROFILE = "Northwind Labs is a logistics marketplace on a mission to make route planning accessible to every retailer."


def test_company_profile_pages_outrank_storefront_ad_copy(settings):
    agent, _, store = _agent(settings, {})
    store.register_official_domain("https://www.northwindlabs.example/")
    ranked = agent._descriptive_hits([
        _hit("https://www.northwindlabs.example/", PROMO),  # official, but it is selling, not describing
        _hit("https://www.linkedin.com/company/northwind-labs", PROFILE),
    ])
    assert [h.url for h in ranked] == ["https://www.linkedin.com/company/northwind-labs", "https://www.northwindlabs.example/"]


def test_promotional_excerpts_are_never_shown_to_the_language_model(settings, plan):
    payload = {"organic_results": [
        {"position": 1, "title": "Northwind Labs", "link": "https://www.northwindlabs.example/", "snippet": PROMO.replace("Grab", "Northwind Labs: grab")},
    ]}
    llm = FakeLLM(["Northwind Labs offers unbeatable prices [E2]."])
    agent, _, _ = _agent(settings, {IDENTITY: payload}, llm)
    p = agent.investigate(OPP, plan.step("identity")).profile
    assert llm.calls == []  # the only excerpt was promotional, so the model was not even asked
    assert p.overview_source == "search_snippet"  # the verbatim excerpt is still shown, clearly labelled as an excerpt
