import json
import re
from datetime import datetime, timezone

import pytest

from agents.investigator import STAGES, Investigator, ProgressEvent
from config import Settings
from models.opportunity import OpportunityProfile
from models.report import InvestigationReport
from services import demo
from services.demo import ReplayClient, demo_client, demo_opportunity, demo_resume, demo_routes
from services.scraping.extractor import extract_from_text
from services.serpapi.client import SerpApiClient
from tests.fakes import FakeLLM
from utils.errors import ScoutLensError, SerpApiError
from utils.language_guard import is_neutral

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings():
    return Settings(_env_file=None, llm_provider="none")


class FaultyReplay(ReplayClient):
    """Replay client that raises a given SerpApiError for chosen (engine, query) pairs."""

    def __init__(self, routes, settings, errors):
        super().__init__(routes, settings)
        self._errors = {(e, q.lower()): err for (e, q), err in errors.items()}

    def _request(self, params, api_key):
        key = (params["engine"], params["q"].lower())
        if key in self._errors:
            self.requested.append(key)
            raise self._errors[key]
        return super()._request(params, api_key)


def run(settings, client=None, *, resume=True, llm=None, opp=None, **kw):
    events: list[ProgressEvent] = []
    client = client or demo_client(NOW, settings)
    report = Investigator(settings, client, llm).run(
        opp or demo_opportunity(NOW), resume=demo_resume() if resume else None, now=NOW, is_demo=True,
        on_progress=events.append, **kw,
    )
    return report, events, client


@pytest.fixture(scope="module")
def full():
    settings = Settings(_env_file=None, llm_provider="none")
    return run(settings)


# ------------------------------------------------------------------ progress

def test_progress_reports_every_stage_in_true_execution_order(full):
    _report, events, _ = full
    finished = [e for e in events if e.status != "running"]
    assert [e.stage for e in finished] == [key for key, _label in STAGES]
    assert [e.index for e in finished] == list(range(1, len(STAGES) + 1))
    assert all(e.status == "done" for e in finished)
    running = [e.stage for e in events if e.status == "running"]
    assert running == [key for key, _ in STAGES if key != "read"]  # "read" completes before the investigator starts


def test_progress_details_come_from_real_results(full):
    _, events, _ = full
    detail = {e.stage: e.detail for e in events if e.status == "done"}
    assert detail["hiring"] == "18 other current listings analysed"
    assert detail["news"] == "3 relevant news items"
    assert detail["market"] == "61 listings sampled"
    assert "follow-up" in detail["followup"] and "funding" in detail["followup"]


def test_a_crashing_progress_callback_does_not_break_the_investigation(settings):
    def boom(_event):
        raise RuntimeError("UI exploded")

    report = Investigator(settings, demo_client(NOW, settings)).run(demo_opportunity(NOW), now=NOW, on_progress=boom)
    assert report.status == "complete"


# ------------------------------------------------------------------- results

def test_the_report_is_complete_and_labelled_demo(full):
    report, _, _ = full
    assert report.status == "complete" and report.is_demo and report.warnings == []
    assert report.opportunity.company_name == "Lumenpath Labs"


def test_company_intelligence(full):
    report, _, _ = full
    company = report.company
    assert company.website == "https://www.lumenpath.example" and company.identity_confirmed
    assert company.overview_source == "knowledge_graph" and company.industry == "Software company"
    assert [f.text for f in company.attributes.values()] == ["Bengaluru, India", "2016"]
    # two from the funding search plus one from a news item, capped at 3; each cites its own source
    assert len(company.funding) == 3 and company.gaps == []
    assert len({tuple(f.evidence_ids) for f in company.funding}) == 3


def test_hiring_signals_are_computed_from_the_collected_listings(full):
    report, _, _ = full
    hiring = report.hiring
    freq = {f.skill: f for f in hiring.skill_frequencies}
    assert hiring.sample_size == 18 and hiring.original_found
    assert (freq["Docker"].count, freq["Docker"].percent) == (7, 38.9)
    assert (freq["AWS"].count, freq["Python"].count) == (10, 15)
    assert not freq["Docker"].in_original and freq["Python"].in_original


def test_experience_variation_matches_the_briefs_example(full):
    report, _, _ = full
    (comp,) = [c for c in report.hiring.comparisons if c.kind == "experience"]
    assert "This listing states 0–2 years" in comp.fact
    assert "1–3 years (2 listings), 2–4 years (1 listing)" in comp.fact


def test_market_signal_matches_the_briefs_docker_example_and_flags_the_resume_gap(full):
    report, _, _ = full
    docker = next(s for s in report.market.signals if s.skill == "Docker")
    assert docker.statement == "23 of 61 sampled listings mention Docker (37.7%)."
    assert docker.resume_status == "not_found" and docker.missing_from_resume is True
    assert docker.interpretation.startswith("Docker appears frequently enough in the sampled listings to consider it a useful preparation area.")
    assert "It was not found on your resume." in docker.interpretation


def test_personal_fit_uses_all_four_statuses_and_shows_its_arithmetic(full):
    report, _, _ = full
    status = {m.skill: m.status for m in report.fit.matches}
    assert status == {"Python": "matched", "SQL": "matched", "REST APIs": "matched", "Git": "matched",
                      "Linux": "not_found", "Machine Learning": "partial"}
    assert report.fit.coverage_percent == 66.7  # 4 of 6
    coverage = next(i for i in report.indicators if i.label == "Resume skill coverage")
    assert coverage.value == "66.7%" and "4 of 6" in coverage.detail


def test_what_you_might_have_missed(full):
    report, _, _ = full
    kinds = [f.kind for f in report.findings]
    assert 3 <= len(report.findings) <= 5
    assert "experience" in kinds and "skills" in kinds and "news" in kinds
    assert "market" in kinds  # Docker: common in the market, absent from this resume
    assert all(f.evidence_ids and f.sources_summary.startswith("Sources: ") for f in report.findings)


def test_crosschecks_and_next_actions(full):
    report, _, _ = full
    status = {c.claim.split(" is ")[0].split(" for ")[0]: c.status for c in report.crosschecks}
    assert status["Official website"] == "corroborated" and status["Funding or investment coverage"] == "corroborated"
    assert any(c.status == "conflicting" and "Conflicting information" in c.detail for c in report.crosschecks)
    texts = [a.text for a in report.next_actions]
    assert texts[0].startswith("Verify the application deadline (2026-10-12)")
    assert any(t.startswith("Prepare Python") for t in texts) and len(texts) <= 8
    assert all(a.reason for a in report.next_actions)


def test_method_section_proves_the_search_activity(full):
    report, _, client = full
    m = report.method
    assert m.searches_performed == 9 == len(client.requested) and m.searches_failed == 0
    assert m.apis_used == ["Google Search", "Google Jobs", "Google News"]
    assert (m.company_listings_analyzed, m.market_listings_analyzed, m.news_results_analyzed) == (18, 61, 4)
    assert m.sources_retained == len(report.evidence) and m.displayed_time == "2026-09-30 17:30 IST"
    assert any("Follow-up search ran" in n for n in m.plan_notes)


def test_indicators_are_independent_measures_not_a_score(full):
    report, _, _ = full
    labels = [i.label for i in report.indicators]
    assert labels == ["Resume skill coverage", "Sampled related listings", "Market sample", "Search results analyzed",
                      "Recent news items", "Evidence items"]
    assert not any("score" in (i.label + i.detail).lower() for i in report.indicators)


# ---------------------------------------------------------- evidence provenance

def _raw_text() -> str:
    return " ".join(json.dumps(list(demo_routes(NOW).values()), ensure_ascii=False).split())


def test_every_cited_url_was_actually_retrieved(full):
    report, _, _ = full
    known = set(re.findall(r"https?://[^\"\s]+", _raw_text())) | {demo.LISTING_URL}
    assert report.evidence, "expected evidence"
    for item in report.evidence:
        assert item.source_url in known, f"invented URL: {item.source_url}"


def test_every_verbatim_excerpt_exists_in_the_retrieved_data(full):
    report, _, _ = full
    raw = _raw_text()
    checked = 0
    for item in report.evidence:
        if item.search_id == "page" or item.source_type == "knowledge_graph":
            continue  # page text is not in the SerpApi payloads; KG excerpts are templated from panel fields
        excerpt = " ".join(item.evidence_text.rstrip("…").split())[:60]
        assert excerpt in raw, f"excerpt not found in retrieved data: {excerpt!r}"
        checked += 1
    assert checked > 50


def test_every_reference_to_evidence_resolves(full):
    report, _, _ = full
    known = {e.id for e in report.evidence}
    refs = [i for f in report.findings for i in f.evidence_ids] + [i for c in report.crosschecks for i in c.evidence_ids]
    refs += [i for a in report.next_actions for i in a.evidence_ids] + report.hiring.listing_evidence_ids
    refs += [i for c in report.hiring.comparisons for i in c.evidence_ids] + [i for s in report.market.signals for i in s.evidence_ids]
    refs += [n.evidence_id for n in report.news.items] + [i for f in report.company.funding for i in f.evidence_ids]
    assert refs and set(refs) <= known


def test_every_source_type_and_tier_is_visible(full):
    report, _, _ = full
    assert {e.tier for e in report.evidence} == {1, 3}  # official site/listings vs aggregators and unclassified sources
    assert all(e.tier_label and e.confidence in ("high", "medium", "low") for e in report.evidence)


def test_report_survives_a_json_round_trip(full):
    report, _, _ = full
    assert InvestigationReport.model_validate_json(report.model_dump_json()) == report


# ------------------------------------------------------------------ language

def _texts(report: InvestigationReport) -> list[str]:
    out = []
    for f in report.findings:
        out += [f.title, f.fact, f.interpretation, f.action or ""]
    for c in report.hiring.comparisons:
        out += [c.fact, c.interpretation, c.action or ""]
    out += report.hiring.notes + report.market.notes + [report.market.method] + report.news.notes + report.fit.notes + report.fit.context
    out += [m.basis for m in report.fit.matches] + report.company.gaps + report.warnings
    for c in report.crosschecks:
        out += [c.claim, c.detail]
    for a in report.next_actions:
        out += [a.text, a.reason]
    for s in report.market.signals:
        out += [s.statement, s.interpretation, s.action or ""]
    for n in report.news.items:
        out += [n.topic, n.relevance_reason]
    return [t for t in out if t]


def test_all_language_written_by_scoutlens_is_neutral(full):
    report, _, _ = full
    texts = _texts(report)
    assert len(texts) > 40
    for text in texts:
        assert is_neutral(text), text


# ------------------------------------------------------------------- failures

def test_missing_api_key_raises_a_setup_error_instead_of_a_partial_report(settings):
    client = SerpApiClient(Settings(_env_file=None))
    with pytest.raises(SerpApiError) as info:
        Investigator(settings, client).run(demo_opportunity(NOW), now=NOW)
    assert info.value.kind == "no_key" and "SERPAPI_KEY" in info.value.user_message


def test_invalid_key_on_the_first_search_is_raised(settings):
    client = FaultyReplay(demo_routes(NOW), settings, {("google", "Lumenpath Labs company"): SerpApiError(kind="invalid_key")})
    with pytest.raises(SerpApiError) as info:
        Investigator(settings, client).run(demo_opportunity(NOW), now=NOW)
    assert info.value.kind == "invalid_key"


def test_quota_exhaustion_midway_yields_a_partial_report_with_what_was_gathered(settings):
    client = FaultyReplay(demo_routes(NOW), settings, {("google_news", "Lumenpath Labs company"): SerpApiError(kind="rate_limited")})
    report, events, _ = run(settings, client)
    status = {e.stage: e.status for e in events if e.status != "running"}
    assert report.status == "partial"
    assert status["identify"] == status["hiring"] == "done"
    assert status["news"] == "failed" and status["followup"] == "skipped" and status["market"] == "skipped"
    assert status["resume"] == "done" and status["correlate"] == "done" and status["report"] == "done"
    assert report.hiring.sample_size == 18 and report.news is None and report.market is None
    assert any("rate limit" in w.lower() and "incomplete" in w.lower() for w in report.warnings)
    assert report.findings and report.fit is not None  # still useful


def test_a_failed_search_makes_the_report_partial_and_says_so(settings):
    client = FaultyReplay(demo_routes(NOW), settings, {("google_jobs", "Software Engineer Intern"): SerpApiError(kind="timeout")})
    report, _, _ = run(settings, client)
    assert report.status == "partial"
    assert report.market.sample_size == 0 and any("failed" in n for n in report.market.notes)
    assert any("searches failed" in w for w in report.warnings) and report.method.searches_failed >= 1


def test_an_unexpected_bug_in_one_stage_is_contained(settings, monkeypatch):
    def broken(*args, **kwargs):
        raise ZeroDivisionError("bug")

    monkeypatch.setattr("agents.investigator.match_skills", broken)
    report, events, _ = run(settings)
    assert report.status == "partial" and report.fit is None
    assert any("failed unexpectedly" in w for w in report.warnings)
    assert all("ZeroDivisionError" not in w and "bug" not in w for w in report.warnings)  # no internals leak to the UI
    assert report.hiring.sample_size == 18  # other stages unaffected


def test_investigation_needs_a_company_name(settings):
    with pytest.raises(ScoutLensError, match="company name"):
        Investigator(settings, demo_client(NOW, settings)).run(OpportunityProfile(role_title="Engineer"), now=NOW)


# ------------------------------------------------------------------- optional inputs

def test_without_a_resume_personal_fit_is_skipped_and_nothing_is_said_about_gaps(settings):
    report, events, _ = run(settings, resume=False)
    assert report.fit is None and report.resume is None
    assert next(e for e in events if e.stage == "resume" and e.status != "running").status == "skipped"
    assert all(s.missing_from_resume is None for s in report.market.signals)
    assert not any(i.label == "Resume skill coverage" for i in report.indicators)


def test_typed_in_skills_alone_still_enable_fit_with_honest_wording(settings):
    from models.preferences import UserPreferences

    report, _, _ = run(settings, resume=False, prefs=UserPreferences(skills=["Python", "Postgres"]))
    assert report.resume.source == "entered"
    status = {m.skill: m for m in report.fit.matches}
    assert status["Python"].status == "matched" and status["SQL"].status == "matched"  # Postgres implies SQL
    assert status["Linux"].status == "unknown" or status["Linux"].basis == "Not found in the skills you entered"


def test_pasted_text_opportunity_is_investigated_without_a_citable_listing(settings):
    text = "Software Engineer Intern\nLocation: Bengaluru\nRequirements:\n- Python, SQL and REST APIs\nApply by 2026-10-12.\n"
    opp = extract_from_text(text, company_name="Lumenpath Labs")
    report, _, _ = run(settings, resume=False, opp=opp)
    assert report.status == "complete"
    assert not any(e.claim == "The supplied listing" for e in report.evidence)
    assert not any(f.kind in ("deadline", "freshness") for f in report.findings)  # nothing citable to base them on
    listing = next(c for c in report.crosschecks if "publicly posted" in c.claim)
    assert listing.status == "insufficient"


# ------------------------------------------------------ evidence gaps trigger follow-ups

def test_thin_first_round_triggers_targeted_follow_up_searches(settings):
    """First round: no knowledge panel, no official-domain result, no company jobs, no news."""
    full_routes = demo_routes(NOW)
    identity = {"organic_results": [
        {"position": 1, "title": "Lumenpath Labs on LinkedIn", "link": "https://www.linkedin.example/company/lumenpath-labs",
         "snippet": "Lumenpath Labs is a software company."},
    ]}
    official_only = json.loads(json.dumps(full_routes[("google", "lumenpath labs company")]))
    del official_only["knowledge_graph"]  # force inference from results, not the panel
    routes = {
        ("google", "lumenpath labs company"): identity,
        ("google", "lumenpath labs official website"): official_only,
        ("google_jobs", "software engineer intern"): full_routes[("google_jobs", "software engineer intern")],
    }
    opp = demo_opportunity(NOW).model_copy(update={"application_url": "https://www.linkedin.com/jobs/view/1", "source_url": "https://www.linkedin.com/jobs/view/1"})
    wide = settings.model_copy(update={"max_follow_up_searches": 4})
    report, events, client = run(wide, ReplayClient(routes, wide), opp=opp)

    detail = next(e.detail for e in events if e.stage == "followup" and e.status != "running")
    assert detail.startswith("Ran 4 follow-up search(es)")
    for name in ("official site", "funding", "jobs fallback", "news fallback"):
        assert name in detail
    assert {("google", "lumenpath labs official website"), ("google", "lumenpath labs funding"),
            ("google_jobs", "lumenpath labs careers"), ("google_news", "lumenpath labs latest")} <= set(client.requested)
    assert any("Follow-up search ran" in n for n in report.method.plan_notes)
    assert report.company.website == "https://www.lumenpath.example/"  # the follow-up established it...
    assert report.company.website_basis.startswith("Inferred")  # ...and says how weak that basis is
    assert "No funding information was found in the retrieved results." in report.company.gaps


def test_follow_up_budget_is_respected_and_the_skipped_search_is_explained(settings):
    routes = {("google", "lumenpath labs company"): {"organic_results": []}}
    opp = demo_opportunity(NOW).model_copy(update={"application_url": "https://www.linkedin.com/jobs/view/1", "source_url": "https://www.linkedin.com/jobs/view/1"})
    report, events, client = run(settings, ReplayClient(routes, settings), opp=opp)
    detail = next(e.detail for e in events if e.stage == "followup" and e.status != "running")
    assert detail.startswith("Ran 3 follow-up search(es)")  # default budget of 3
    assert any("budget reached" in n for n in report.method.plan_notes)


# ---------------------------------------------------------------------- LLM

def test_unavailable_llm_is_reported_and_the_run_continues(settings):
    ollama = settings.model_copy(update={"llm_provider": "ollama"})
    report, _, _ = run(ollama, llm=FakeLLM(available=False))
    assert report.status == "complete" and report.method.llm is None
    assert any("Language model not available" in w and "deterministic" in w for w in report.warnings)


def test_llm_disabled_by_choice_is_not_a_warning(settings):
    report, _, _ = run(settings, llm=FakeLLM(available=False))  # llm_provider == "none"
    assert report.warnings == []


def test_llm_contributions_are_recorded_when_used(settings):
    judgement = json.dumps({"items": [{"index": 0, "relevance": "high", "why": "Describes a product launch tied to engineering."}]})
    report, _, _ = run(settings, llm=FakeLLM([judgement]))
    assert report.method.llm == "fake" and report.method.llm_used_for == ["news relevance"]
    assert report.news.items[0].relevance_reason == "Describes a product launch tied to engineering."


def test_refresh_flag_is_recorded(settings):
    report, _, _ = run(settings, refresh=True)
    assert report.method.refreshed is True
