import socket
from datetime import datetime, timezone

import pytest
import responses

from agents.investigator import STAGES
from config import Settings
from models.preferences import UserPreferences
from services import demo
from services.demo import ReplayClient, demo_routes, listing_page_html
from services.pipeline import InvestigationRequest, NeedsCompanyError, run_pipeline
from services.scraping import fetcher as fetcher_module
from services.scraping.fetcher import PageFetcher
from tests.pdf_factory import make_pdf
from utils.errors import ExtractionError, InvalidURLError

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
URL = demo.LISTING_URL
ROBOTS = "https://www.lumenpath.example/robots.txt"


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))])
    fetcher_module._last_request.clear()


@pytest.fixture
def settings():
    return Settings(_env_file=None, llm_provider="none")


@pytest.fixture
def replay(settings):
    return ReplayClient(demo_routes(NOW), settings)


def fetcher():
    return PageFetcher(sleep=lambda _s: None)


def serve(page_html: str, url: str = URL):
    responses.add(responses.GET, ROBOTS, status=404)
    responses.add(responses.GET, url, body=page_html, content_type="text/html")


# ------------------------------------------------------------------- URL flow

@responses.activate
def test_url_flow_reads_the_page_then_investigates(settings, replay):
    serve(listing_page_html(NOW))
    events = []
    report = run_pipeline(InvestigationRequest(url=URL), settings, client=replay, fetcher=fetcher(), now=NOW, on_progress=events.append)

    assert report.is_demo is False and report.status == "complete"
    assert (report.opportunity.company_name, report.opportunity.role_title) == ("Lumenpath Labs", "Software Engineer Intern")
    assert report.hiring.sample_size == 18
    finished = [e for e in events if e.status != "running"]
    assert [e.stage for e in finished] == [k for k, _ in STAGES] and finished[0].detail == "Software Engineer Intern at Lumenpath Labs"
    assert events[0].stage == "read" and events[0].status == "running"  # the read stage is reported once, and first
    assert sum(e.stage == "read" for e in events) == 2
    listing = [e for e in report.evidence if e.claim == "The supplied listing"]
    assert len(listing) == 1 and listing[0].source_url == URL  # the page that was really fetched is citable


@responses.activate
def test_missing_or_bad_url_is_rejected_before_anything_is_fetched(settings, replay):
    with pytest.raises(InvalidURLError):
        run_pipeline(InvestigationRequest(url="javascript:alert(1)"), settings, client=replay, fetcher=fetcher())
    with pytest.raises(InvalidURLError, match="Paste an opportunity URL"):
        run_pipeline(InvestigationRequest(), settings, client=replay, fetcher=fetcher())
    assert len(responses.calls) == 0


@responses.activate
def test_unreadable_page_reports_failure_and_suggests_pasting(settings, replay):
    serve((__import__("pathlib").Path(__file__).parent / "fixtures" / "job_js_only.html").read_text(encoding="utf-8"))
    events = []
    with pytest.raises(ExtractionError) as info:
        run_pipeline(InvestigationRequest(url=URL), settings, client=replay, fetcher=fetcher(), on_progress=events.append)
    assert "paste" in info.value.user_message.lower()
    assert [(e.stage, e.status) for e in events] == [("read", "running"), ("read", "failed")]


# --------------------------------------------------------------- manual fallback

PASTED = "Software Engineer Intern\nLocation: Bengaluru\nRequirements:\n- Python, SQL and REST APIs\n- 0-2 years of experience\nNice to have:\n- Git\n"


def test_pasted_description_with_a_company_is_investigated(settings, replay):
    report = run_pipeline(InvestigationRequest(pasted_text=PASTED, company="Lumenpath Labs"), settings, client=replay, now=NOW)
    assert report.opportunity.extraction_method == "manual" and report.opportunity.field_sources["company_name"] == "manual"
    assert report.opportunity.required_skills == ["Python", "SQL", "REST APIs"]
    assert report.hiring.sample_size == 18 and report.status == "complete"


def test_pasted_description_without_a_company_asks_for_it_and_keeps_what_was_extracted(settings, replay):
    with pytest.raises(NeedsCompanyError) as info:
        run_pipeline(InvestigationRequest(pasted_text=PASTED), settings, client=replay)
    assert info.value.opportunity.required_skills == ["Python", "SQL", "REST APIs"]
    assert "company name" in info.value.user_message.lower()

    # the UI then re-submits the already-read opportunity plus the company the user typed
    request = InvestigationRequest(opportunity=info.value.opportunity, company="Lumenpath Labs")
    report = run_pipeline(request, settings, client=replay, now=NOW)
    assert report.opportunity.company_name == "Lumenpath Labs" and report.opportunity.field_sources["company_name"] == "manual"


@responses.activate
def test_a_page_without_a_company_offers_the_partial_result_instead_of_failing(settings, replay):
    html = ("<html><head><title>Backend Engineer</title></head><body><main>" + "<p>Build APIs in Python and SQL for the team every day.</p>" * 12 + "</main></body></html>")
    serve(html, "https://www.linkedin.com/jobs/view/1")
    responses.add(responses.GET, "https://www.linkedin.com/robots.txt", status=404)
    with pytest.raises(NeedsCompanyError) as info:
        run_pipeline(InvestigationRequest(url="https://www.linkedin.com/jobs/view/1"), settings, client=replay, fetcher=fetcher())
    assert info.value.opportunity.role_title == "Backend Engineer" and {"Python", "SQL"} <= set(info.value.opportunity.required_skills)


def test_user_typed_fields_override_extracted_ones_and_are_marked_manual(settings, replay):
    request = InvestigationRequest(pasted_text=PASTED, company="Lumenpath Labs", role="Platform Intern", location="Pune")
    report = run_pipeline(request, settings, client=replay, now=NOW)
    opp = report.opportunity
    assert (opp.role_title, opp.location) == ("Platform Intern", "Pune")
    assert opp.field_sources["role_title"] == opp.field_sources["location"] == "manual"


# --------------------------------------------------------------------- resume

RESUME_LINES = (
    "ASHA VERMA|Skills|Languages: Python, SQL, JavaScript|Frameworks: FastAPI, React|Databases: Postgres, MongoDB|Tools: Git, Docker|"
    "Projects|Tracker | Python, Flask|- Built a REST API with Flask and PostgreSQL|Experience|Backend Intern - Acme|- Wrote SQL reports in Python"
).split("|")


def test_a_pdf_resume_is_parsed_and_compared(settings, replay):
    request = InvestigationRequest(pasted_text=PASTED, company="Lumenpath Labs", resume_bytes=make_pdf(RESUME_LINES))
    events = []
    report = run_pipeline(request, settings, client=replay, now=NOW, on_progress=events.append)
    assert report.resume.name == "Asha Verma" and report.resume.source == "pdf"
    assert {m.skill: m.status for m in report.fit.matches}["Python"] == "matched"
    assert "resume parsed" in next(e.detail for e in events if e.stage == "read" and e.status == "done")


def test_an_unusable_resume_does_not_block_the_investigation(settings, replay):
    request = InvestigationRequest(pasted_text=PASTED, company="Lumenpath Labs", resume_bytes=b"this is not a pdf")
    report = run_pipeline(request, settings, client=replay, now=NOW)
    assert report.resume is None and report.fit is None
    assert report.status == "partial" and "not a PDF" in report.warnings[0] and "personal-fit analysis was skipped" in report.warnings[0]
    assert report.hiring.sample_size == 18  # everything else still ran


def test_an_already_parsed_resume_is_reused_and_preferences_are_remembered(settings, replay):
    from services.resume.parser import parse_resume_text

    profile = parse_resume_text("\n".join(RESUME_LINES))
    prefs = UserPreferences(location="Pune", skills=["Kubernetes"])
    report = run_pipeline(InvestigationRequest(pasted_text=PASTED, company="Lumenpath Labs", resume_profile=profile, prefs=prefs), settings, client=replay, now=NOW)
    assert report.resume.skill("Kubernetes").level == "self_reported"
    assert report.preferences == prefs
    assert report.market.location == "Pune"  # the location preference steers the market sample


# ------------------------------------------------------------------------ demo

def test_demo_needs_no_key_no_network_and_no_llm(settings):
    assert not settings.has_serpapi_key
    report = run_pipeline(InvestigationRequest(demo=True), settings, now=NOW)
    assert report.is_demo and report.status == "complete" and report.method.llm is None
    assert report.opportunity.company_name == "Lumenpath Labs" and report.fit is not None
    assert report.method.searches_performed == 9


def test_refresh_flag_flows_through(settings, replay):
    report = run_pipeline(InvestigationRequest(pasted_text=PASTED, company="Lumenpath Labs", refresh=True), settings, client=replay, now=NOW)
    assert report.method.refreshed is True
