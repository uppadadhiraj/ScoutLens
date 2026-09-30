import pytest

from agents.market_agent import MarketAgent
from agents.planner import InvestigationPlanner
from models.opportunity import OpportunityProfile
from models.search import JobListing
from services.analysis.market import analyze_market, interpret
from services.evidence.store import EvidenceStore
from services.resume.parser import parse_resume_text
from tests.fakes import ScriptedClient
from utils.errors import SerpApiError
from utils.language_guard import is_neutral

ROLE = "Software Engineer Intern"
OPP = OpportunityProfile(
    company_name="Northwind Labs", role_title=ROLE, required_skills=["Python", "Docker"], preferred_skills=["Kubernetes"],
    source_url="https://www.northwindlabs.example/careers/swe-intern",
)
RESUME = parse_resume_text("""Asha Verma

Skills
Languages: Python, SQL, JavaScript
Frameworks: FastAPI, React
Databases: PostgreSQL, MongoDB
Tools: Git, Linux

Projects
Tracker | Flask
• Built a REST API with Flask and PostgreSQL
• Wrote unit tests with pytest and deployed on a Linux server for a small user group

Experience
Intern - Acme Corp
• Built dashboards and automation scripts in Python for the analytics team
""")


RESUME_TEXT_WITH_AZURE = """Asha Verma

Skills
Languages: Python, SQL, JavaScript
Frameworks: FastAPI, React
Databases: PostgreSQL, MongoDB
Tools: Git, Azure

Projects
Tracker | Flask
• Built a REST API with Flask and PostgreSQL and deployed it on Azure
"""


def listing(n, desc, title=ROLE, company="Some Co"):
    return JobListing(
        search_id="s1", job_id=f"m{n}", title=f"{title}", company_name=f"{company} {n}", location="Bengaluru, India",
        description=desc, share_link=f"https://www.google.com/search?q=job{n}", via="LinkedIn",
    )


def sample(n_docker, n_python, total, n_k8s=0):
    """`total` listings; the first n_docker mention Docker, the first n_python mention Python, etc."""
    out = []
    for i in range(total):
        words = []
        if i < n_docker:
            words.append("Docker")
        if i < n_python:
            words.append("Python")
        if i < n_k8s:
            words.append("Kubernetes")
        out.append(listing(i, f"We use {' and '.join(words) or 'internal tools'} daily."))
    return out


def run(listings, opp=OPP, resume=None, **kw):
    store = EvidenceStore(opp.company_name)
    result = analyze_market(opp, listings, store, role_query=ROLE, location="India", resume=resume, **kw)
    return result, store


def sig(result, skill):
    return next(s for s in result.signals if s.skill == skill)


def test_percentages_use_the_real_sample_size():
    result, _ = run(sample(n_docker=23, n_python=40, total=61))
    docker = sig(result, "Docker")
    assert (docker.count, docker.total, docker.percent) == (23, 61, 37.7)
    assert docker.statement == "23 of 61 sampled listings mention Docker (37.7%)."
    assert result.sample_size == 61 and not result.low_sample


def test_wording_matches_the_brief_and_never_commands_the_user():
    result, _ = run(sample(23, 40, 61))
    docker = sig(result, "Docker")
    assert docker.interpretation == "Docker appears frequently enough in the sampled listings to consider it a useful preparation area."
    for s in result.signals:
        text = f"{s.statement} {s.interpretation} {s.action or ''}"
        assert is_neutral(text) and "must learn" not in text.lower() and "you should" not in text.lower()


@pytest.mark.parametrize(
    "percent, fragment",
    [(37.7, "frequently enough"), (30.0, "frequently enough"), (29.9, "minority"), (10.0, "minority"), (9.9, "few of the sampled"), (0.0, "few of the sampled")],
)
def test_interpretation_thresholds(percent, fragment):
    assert fragment in interpret("Docker", percent, None)[0]


def test_resume_gap_is_flagged_and_only_a_frequent_gap_gets_an_action():
    result, _ = run(sample(23, 40, 61), resume=RESUME)
    docker, python = sig(result, "Docker"), sig(result, "Python")
    assert docker.missing_from_resume is True and "It was not found on your resume." in docker.interpretation
    assert docker.action == "Review Docker fundamentals if this fits your goals."
    assert python.missing_from_resume is False and python.action is None
    assert result.signals[0].missing_from_resume is True  # gaps are listed first


def test_infrequent_gap_has_no_action():
    result, _ = run(sample(n_docker=2, n_python=40, total=61), resume=RESUME)
    assert sig(result, "Docker").action is None


def test_without_a_resume_nothing_is_said_about_gaps():
    result, _ = run(sample(23, 40, 61))
    assert all(s.missing_from_resume is None for s in result.signals)
    assert not any("resume" in s.interpretation for s in result.signals)


def test_adjacent_skills_are_common_in_the_sample_but_not_named_by_the_listing():
    listings = [listing(i, "Python with AWS and Terraform") for i in range(10)] + [listing(i + 10, "Python only") for i in range(10)]
    result, _ = run(listings)
    adjacent = {s.skill: s for s in result.signals if s.scope == "adjacent"}
    assert set(adjacent) == {"AWS", "Terraform"} and adjacent["AWS"].percent == 50.0
    assert {s.skill for s in result.signals if s.scope == "in_listing"} == {"Python", "Docker", "Kubernetes"}
    assert sig(result, "Docker").count == 0  # named by the listing, never seen in the sample: reported as 0, not hidden


def test_adjacent_skills_are_compared_with_the_resume_too():
    listings = [listing(i, "Python with AWS and Terraform") for i in range(10)] + [listing(i + 10, "Python only") for i in range(10)]
    result, _ = run(listings, resume=RESUME)
    aws = sig(result, "AWS")
    assert aws.scope == "adjacent" and aws.resume_status == "not_found" and aws.missing_from_resume is True
    assert "It was not found on your resume." in aws.interpretation


def test_a_related_skill_is_reported_as_partial_not_as_not_found():
    with_azure = parse_resume_text(RESUME_TEXT_WITH_AZURE)
    listings = [listing(i, "Python with AWS") for i in range(12)]
    result, _ = run(listings, resume=with_azure)
    aws = sig(result, "AWS")
    assert aws.resume_status == "partial" and aws.missing_from_resume is True
    assert "Only a related skill or a passing mention was found on your resume." in aws.interpretation
    assert "was not found" not in aws.interpretation
    assert aws.action == "Review AWS fundamentals if this fits your goals."


def test_unknown_resume_status_makes_no_claim_about_the_resume():
    sparse = parse_resume_text("Jane Roe\n\nSkills\nPython\n")
    result, _ = run(sample(23, 40, 61), resume=sparse)
    docker = sig(result, "Docker")
    assert docker.resume_status == "unknown" and docker.missing_from_resume is None
    assert "resume" not in docker.interpretation and docker.action is None


def test_the_original_listing_and_offrole_titles_are_excluded_from_the_sample():
    original = listing(900, "Docker Python", company="Northwind Labs")
    original.apply_links = []
    original.share_link = "https://www.northwindlabs.example/careers/swe-intern"
    off_role = listing(901, "Docker everywhere", title="Account Executive")
    result, _ = run([original, off_role, *sample(5, 5, 10)])
    assert result.sample_size == 10 and result.excluded_off_role == 1
    assert any("unrelated titles" in n for n in result.notes)


def test_duplicate_listings_are_counted_once():
    result, _ = run(sample(3, 3, 10) + sample(3, 3, 10))
    assert result.sample_size == 10


def test_small_samples_are_flagged_not_hidden():
    result, _ = run(sample(2, 3, 5))
    assert result.low_sample and any("Small sample (5 listings)" in n for n in result.notes)
    assert sig(result, "Docker").total == 5


def test_no_comparable_listings_is_insufficient_evidence_not_invented_numbers():
    result, _ = run([])
    assert result.signals == [] and any(n.startswith("Insufficient evidence") for n in result.notes)


def test_method_statement_is_transparent():
    result, _ = run(sample(3, 3, 12))
    assert "returned by Google Jobs for 'Software Engineer Intern' in India" in result.method
    assert "not a random sample" in result.method and "role title only" in result.method


def test_signals_cite_the_listings_that_mention_the_skill():
    result, store = run(sample(n_docker=4, n_python=6, total=12))
    docker = sig(result, "Docker")
    assert len(docker.evidence_ids) == 4
    assert all(store.get(i).source_type == "job" for i in docker.evidence_ids)
    assert len(sig(result, "Python").evidence_ids) == 6


# ------------------------------------------------------------------------ agent

@pytest.fixture
def market_step(settings):
    return InvestigationPlanner(settings).initial_plan(OPP).step("market")


def test_agent_runs_the_role_level_search_and_records_it(settings, market_step):
    pages = [{"jobs_results": [_raw(i) for i in range(0, 6)]}, {"jobs_results": [_raw(i) for i in range(6, 12)]}]
    client = ScriptedClient(settings, {("google_jobs", ROLE): pages})
    analysis = MarketAgent(client, EvidenceStore(OPP.company_name), settings.serpapi_location).investigate(OPP, market_step)
    assert analysis.sample_size == 12 and analysis.location == "India"
    assert analysis.search_ids == ["s1", "s2", "s3"][:len(client.records)] and len(client.records) == 2
    assert client.requested == [("google_jobs", ROLE.lower())] * 2
    assert "docker" not in market_step.query.q.lower()


def _raw(i):
    return {"title": ROLE, "company_name": f"Co {i}", "job_id": f"r{i}", "description": "Docker and Python" if i % 2 else "Excel",
            "share_link": f"https://www.google.com/search?q=r{i}"}


def test_agent_reports_a_failed_sample_without_crashing(settings, market_step):
    client = ScriptedClient(settings, errors={("google_jobs", ROLE): SerpApiError(kind="timeout")})
    analysis = MarketAgent(client, EvidenceStore(OPP.company_name)).investigate(OPP, market_step)
    assert analysis.sample_size == 0 and any("failed" in n for n in analysis.notes)


def test_agent_propagates_fatal_errors(settings, market_step):
    client = ScriptedClient(settings, errors={("google_jobs", ROLE): SerpApiError(kind="rate_limited")})
    with pytest.raises(SerpApiError):
        MarketAgent(client, EvidenceStore(OPP.company_name)).investigate(OPP, market_step)


def test_skills_already_on_the_resume_are_not_called_a_preparation_area():
    text, action = interpret("Python", 72.1, "matched")
    assert text == "Python appears frequently in the sampled listings, and it is on your resume." and action is None
    assert "preparation area" not in text
    assert "preparation area" in interpret("Docker", 37.7, "not_found")[0]
