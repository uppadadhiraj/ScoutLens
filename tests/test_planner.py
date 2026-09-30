import json

import pytest

from agents.planner import InvestigationPlanner, simplify_role
from config import Settings
from models.opportunity import OpportunityProfile
from models.plan import PlanState
from models.preferences import UserPreferences
from tests.fakes import FakeLLM


def _opp(role="Software Engineer Intern (Backend) - Pune", company="Northwind Labs") -> OpportunityProfile:
    return OpportunityProfile(company_name=company, role_title=role)


def _planner(**kw) -> InvestigationPlanner:
    llm = kw.pop("llm", None)
    return InvestigationPlanner(Settings(_env_file=None, **kw), llm=llm)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Software Engineer Intern (Backend) - Pune", "Software Engineer Intern"),
        ("Data Analyst, Growth", "Data Analyst"),
        ("Backend Engineer | Remote", "Backend Engineer"),
        ("SDE", "SDE"),
    ],
)
def test_simplify_role(raw, expected):
    assert simplify_role(raw) == expected


def test_initial_plan_covers_identity_jobs_news_and_market():
    plan = _planner().initial_plan(_opp())
    assert [s.key for s in plan.searches] == ["identity", "jobs_company", "jobs_role", "news", "market"]
    by_key = {s.key: s.query for s in plan.searches}
    assert (by_key["identity"].engine, by_key["identity"].q) == ("google", "Northwind Labs company")
    assert by_key["jobs_company"].q == "Northwind Labs jobs" and by_key["jobs_company"].pages == 2
    assert by_key["jobs_role"].q == "Software Engineer Intern Northwind Labs"
    assert (by_key["news"].engine, by_key["news"].q) == ("google_news", "Northwind Labs company")
    assert by_key["market"].q == "Software Engineer Intern" and by_key["market"].pages == 3


def test_market_sample_never_contains_the_skills_it_measures():
    opp = _opp()
    opp.required_skills = ["Docker", "AWS"]
    market = _planner().initial_plan(opp).step("market").query
    assert "docker" not in market.q.lower() and "aws" not in market.q.lower()


def test_preferences_steer_the_market_sample():
    prefs = UserPreferences(target_role="Data Engineer", location="Pune, Maharashtra, India")
    market = _planner().initial_plan(_opp(), prefs).step("market").query
    assert market.q == "Data Engineer" and market.location == "Pune, Maharashtra, India"


def test_no_role_means_no_role_specific_searches():
    plan = _planner().initial_plan(_opp(role=None))
    assert [s.key for s in plan.searches] == ["identity", "jobs_company", "news"]


def test_planner_requires_a_company():
    with pytest.raises(ValueError):
        _planner().initial_plan(_opp(company=None))


def test_page_budgets_come_from_settings():
    plan = _planner(company_job_pages=1, market_pages=2).initial_plan(_opp())
    assert plan.step("jobs_company").query.pages == 1 and plan.step("market").query.pages == 2


# -------------------------------------------------------------------- follow-ups

def test_follow_ups_fill_every_gap_within_budget():
    state = PlanState()  # nothing found at all
    steps, notes = _planner().follow_ups(_opp(), state, executed=set())
    assert [s.key for s in steps] == ["official_site", "funding", "jobs_fallback"]  # budget of 3
    assert any("budget" in n for n in notes)  # news_fallback was deferred, and says so


def test_follow_ups_are_empty_when_nothing_is_missing():
    state = PlanState(official_domain_found=True, funding_searched=True, company_job_count=5, news_count=3, has_knowledge_graph=True)
    assert _planner().follow_ups(_opp(), state, set()) == ([], [])


def test_established_company_skips_funding_and_says_why():
    state = PlanState(official_domain_found=True, company_job_count=5, news_count=3, kg_founded_year=1995)
    steps, notes = _planner().follow_ups(_opp(), state, set())
    assert steps == [] and any("established (founded 1995)" in n for n in notes)


def test_young_company_gets_a_funding_search():
    state = PlanState(official_domain_found=True, company_job_count=5, news_count=3, kg_founded_year=2016)
    steps, _ = _planner().follow_ups(_opp(), state, set())
    assert [s.query.q for s in steps] == ["Northwind Labs funding"]


def test_follow_ups_never_repeat_an_executed_search():
    state = PlanState(official_domain_found=True, company_job_count=5, news_count=3)
    steps, _ = _planner().follow_ups(_opp(), state, executed={("google", "northwind labs funding")})
    assert steps == []


# -------------------------------------------------------------------- LLM extras

def _extras(*queries) -> str:
    return json.dumps({"queries": [{"engine": e, "q": q, "reason": "r"} for e, q in queries]})


def test_llm_extras_are_off_by_default():
    llm = FakeLLM([_extras(("google", "Northwind Labs engineering blog"))])
    plan = _planner(llm=llm).initial_plan(_opp())
    assert len(plan.searches) == 5 and llm.calls == []


def test_llm_extras_are_validated_deduplicated_and_capped():
    reply = _extras(
        ("google", "Northwind Labs engineering blog"),
        ("google", "best pizza near me"),  # not about the company -> rejected
        ("google_news", "northwind labs company"),  # duplicate of the news search -> rejected
        ("google_news", "Northwind Labs partnerships"),
        ("google", "Northwind Labs glassdoor reviews"),  # over the cap of 2
    )
    plan = _planner(llm_planning=True, llm=FakeLLM([reply])).initial_plan(_opp())
    extras = [s for s in plan.searches if s.key.startswith("llm_extra")]
    assert [s.query.q for s in extras] == ["Northwind Labs engineering blog", "Northwind Labs partnerships"]
    assert any("planning model" in n for n in plan.notes)


def test_llm_failure_never_breaks_planning():
    plan = _planner(llm_planning=True, llm=FakeLLM(available=False)).initial_plan(_opp())
    assert len(plan.searches) == 5
    plan = _planner(llm_planning=True, llm=FakeLLM(["garbage"] * 3)).initial_plan(_opp())
    assert len(plan.searches) == 5


# ---- where to search (a hardcoded country compared a Stockholm role with Mumbai listings) ----------

from agents.planner import search_location  # noqa: E402


@pytest.mark.parametrize(
    "listing_location, pref_location, expected",
    [
        ("Stockholm; London", None, "Stockholm"),
        ("Bengaluru, Karnataka, India", None, "Bengaluru, Karnataka, India"),
        ("Stockholm; London", "Pune, India", "Pune, India"),  # the user's preference wins
        ("Remote", None, None),
        ("Anywhere", None, None),
        (None, None, None),
        ("  ", None, None),
    ],
)
def test_search_location(listing_location, pref_location, expected):
    opp = OpportunityProfile(company_name="Acme", role_title="Engineer", location=listing_location)
    assert search_location(opp, UserPreferences(location=pref_location)) == expected


def test_job_searches_use_the_listings_own_location_and_identity_asks_for_the_company():
    opp = OpportunityProfile(company_name="Northwind Labs", role_title="Android Engineer", location="Stockholm; London")
    plan = _planner().initial_plan(opp)
    for key in ("jobs_company", "jobs_role", "market"):
        assert plan.step(key).query.location == "Stockholm", key
    assert plan.step("identity").query.q == "Northwind Labs company"  # a bare brand name returns songs and login pages
    assert plan.step("news").query.q == "Northwind Labs company"
    assert plan.step("identity").query.location is None  # identity/news are not location-scoped
