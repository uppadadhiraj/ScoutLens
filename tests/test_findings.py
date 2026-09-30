from datetime import date

import pytest

from models.company import CompanyFact, CompanyProfile
from models.opportunity import OpportunityProfile
from models.report import Finding, HiringSignal, MarketAnalysis, MarketSignal, NewsHighlight, NewsSignal, SignalComparison
from models.search import SearchHit
from services.analysis.findings import build_findings, parse_listing_date, select_findings, summarize_sources
from services.evidence.store import EvidenceStore
from utils.language_guard import is_neutral

CO = "Northwind Labs"
TODAY = date(2026, 9, 30)


def store_with(n_jobs=0, n_news=0):
    """A store holding real EvidenceItems (created through the public factories), plus their ids."""
    from models.search import JobListing, NewsItem

    store = EvidenceStore(CO)
    jobs = [store.add_job(JobListing(search_id="s1", title=f"Job {i}", company_name=CO, share_link=f"https://g.example/j{i}"), f"opening {i}").id for i in range(n_jobs)]
    news = [store.add_news(NewsItem(search_id="s2", title=f"Northwind story {i}", url=f"https://n.example/{i}", publisher="Reuters"), f"news {i}").id for i in range(n_news)]
    return store, jobs, news


def comp(kind, fact, ids, strength=0.5, **kw):
    return SignalComparison(kind=kind, fact=fact, interpretation="A neutral interpretation.", evidence_ids=ids, strength=strength, **kw)


def hiring(comparisons):
    return HiringSignal(company=CO, sample_size=5, comparisons=comparisons)


OPP = OpportunityProfile(company_name=CO, role_title="Software Engineer Intern", source_url="https://www.northwindlabs.example/careers/x")


# --------------------------------------------------------------------- dates

@pytest.mark.parametrize(
    "text, expected",
    [("2026-10-31", date(2026, 10, 31)), ("2026-10-31T23:59", date(2026, 10, 31)), ("15 October 2026", date(2026, 10, 15)),
     ("15th October 2026", date(2026, 10, 15)), ("Oct 31, 2026", date(2026, 10, 31)), ("1 Sept 2026", date(2026, 9, 1)),
     ("03/04/2026", None), ("soon", None), (None, None)],
)
def test_parse_listing_date(text, expected):
    assert parse_listing_date(text) == expected


# ------------------------------------------------------------ source summaries

def test_summarize_sources_pluralises_and_counts_by_type():
    store, jobs, news = store_with(18, 1)
    assert summarize_sources(jobs, store) == "Sources: 18 job listings"
    assert summarize_sources(jobs[:1] + news, store) == "Sources: 1 job listing, 1 news article"
    assert summarize_sources([], store) == "Sources: none"


# --------------------------------------------------------------- building

def test_finding_facts_and_evidence_come_straight_from_the_analysis():
    store, jobs, _ = store_with(7)
    fact = "Docker appears in 7 of 18 sampled current listings from Northwind Labs (38.9%) but is not mentioned in this listing."
    findings = build_findings(OPP, None, hiring([comp("skills", fact, jobs, 0.75, action="Check whether Docker is used on this team.")]), None, None, store, today=TODAY)
    (f,) = findings
    assert (f.kind, f.icon, f.title) == ("skills", "🔎", "Hiring Pattern")
    assert f.fact == fact and f.evidence_ids == jobs and f.sources_summary == "Sources: 7 job listings"
    assert f.action == "Check whether Docker is used on this team."


def test_a_candidate_without_resolvable_evidence_is_dropped():
    store, jobs, _ = store_with(2)
    findings = build_findings(OPP, None, hiring([
        comp("skills", "uncited claim", [], 0.99),
        comp("volume", "phantom citation", ["E999"], 0.98),
        comp("location", "cited claim", jobs[:1], 0.4),
    ]), None, None, store, today=TODAY)
    assert [f.fact for f in findings] == ["cited claim"]


def test_the_finding_model_itself_refuses_empty_evidence():
    with pytest.raises(ValueError):
        Finding(kind="skills", icon="x", title="t", fact="f", interpretation="i", evidence_ids=[], sources_summary="")


def test_at_most_five_findings_at_most_two_per_kind():
    store, jobs, _ = store_with(10)
    many = [comp("skills", f"skill fact {i}", [jobs[i]], 0.9 - i * 0.01) for i in range(6)] + [
        comp(k, f"{k} fact", [jobs[7]], 0.5) for k in ("volume", "experience", "location", "salary")
    ]
    findings = build_findings(OPP, None, hiring(many), None, None, store, today=TODAY)
    assert len(findings) == 5
    kinds = [f.kind for f in findings]
    assert kinds.count("skills") <= 2
    assert [f.strength for f in findings] == sorted((f.strength for f in findings), reverse=True)


def test_hiring_cards_leave_room_for_news_and_market_evidence():
    store, jobs, news = store_with(8, 1)
    many = [comp(k, f"{k} fact", [jobs[i]], 0.9 - i * 0.02) for i, k in enumerate(["skills", "experience", "location", "volume", "salary", "work_mode"])]
    item = NewsHighlight(title="Northwind Labs launches route planner", url="https://n.example/0", publisher="Reuters", date_text="3 days ago",
                         summary="s", topic="Product & launches", relevance="medium", relevance_reason="Company-level coverage on the topic: product & launches.",
                         evidence_id=news[0], tier=2)
    market = MarketAnalysis(role_query="SWE", sample_size=61, signals=[MarketSignal(
        skill="Docker", count=23, total=61, scope="adjacent", statement="23 of 61 sampled listings mention Docker (37.7%).",
        interpretation="Docker appears frequently enough in the sampled listings to consider it a useful preparation area.", evidence_ids=jobs[:3])])
    findings = build_findings(OPP, None, hiring(many), NewsSignal(company=CO, items=[item]), market, store, today=TODAY)
    kinds = {f.kind for f in findings}
    assert {"news", "market"} <= kinds and len(findings) == 5
    assert sum(k in {"skills", "experience", "location", "volume", "salary", "work_mode"} for k in [f.kind for f in findings]) == 3


def test_when_there_is_little_else_hiring_findings_fill_the_slots():
    store, jobs, _ = store_with(6)
    cands = [comp(k, f"{k} fact", [jobs[i]], 0.6 - i * 0.01) for i, k in enumerate(["skills", "experience", "location", "volume", "salary"])]
    assert len(build_findings(OPP, None, hiring(cands), None, None, store, today=TODAY)) == 5


def test_news_finding_quotes_the_headline_and_keeps_the_interpretation_separate():
    store, _, news = store_with(0, 1)
    item = NewsHighlight(title="Northwind Labs raises Series B", url="https://n.example/0", publisher="Reuters", date_text="09/29/2026",
                         summary="s", topic="Funding & investment", relevance="high", relevance_reason="Mentions 'engineer', which appears in this role's title or requirements.",
                         evidence_id=news[0], tier=2)
    (f,) = build_findings(OPP, None, None, NewsSignal(company=CO, items=[item]), None, store, today=TODAY)
    assert f.fact == "Reuters · 09/29/2026: “Northwind Labs raises Series B”"
    assert f.interpretation.startswith("Mentions 'engineer'") and f.evidence_ids == news


def test_low_relevance_news_is_not_a_finding():
    store, _, news = store_with(0, 1)
    item = NewsHighlight(title="t", url="https://n.example/0", summary="s", topic="General coverage", relevance="low", relevance_reason="r", evidence_id=news[0], tier=2)
    assert build_findings(OPP, None, None, NewsSignal(company=CO, items=[item]), None, store, today=TODAY) == []


def test_funding_finding_is_flagged_as_unverified_by_scoutlens():
    store = EvidenceStore(CO)
    hit = store.add_hit(SearchHit(search_id="s1", title="Northwind raises", url="https://n.example/f", snippet="Northwind Labs raised a seed round."), "funding")
    company = CompanyProfile(name=CO, funding=[CompanyFact(text="Northwind Labs raised a seed round.", evidence_ids=[hit.id])])
    (f,) = build_findings(OPP, company, None, None, None, store, today=TODAY)
    assert f.kind == "funding" and "not independently verified" in f.interpretation


def _market(resume_gap: bool):
    return MarketAnalysis(role_query="SWE", sample_size=61, signals=[
        MarketSignal(skill="Docker", count=23, total=61, scope="adjacent", missing_from_resume=True if resume_gap else None,
                     statement="23 of 61 sampled listings mention Docker (37.7%).", interpretation="Docker appears frequently enough in the sampled listings to consider it a useful preparation area.",
                     action="Review Docker fundamentals if this fits your goals.", evidence_ids=["E1"]),
        MarketSignal(skill="Excel", count=2, total=61, scope="adjacent", statement="2 of 61 sampled listings mention Excel (3.3%).", interpretation="x", evidence_ids=["E1"]),
    ])


def test_market_finding_needs_a_resume_gap_when_a_resume_was_supplied():
    store, _, _ = store_with(2)
    (f,) = build_findings(OPP, None, None, None, _market(resume_gap=True), store, resume_supplied=True, today=TODAY)
    assert f.kind == "market" and f.action == "Review Docker fundamentals if this fits your goals."
    no_gap = _market(resume_gap=False)
    assert build_findings(OPP, None, None, None, no_gap, store, resume_supplied=True, today=TODAY) == []


def test_without_a_resume_a_frequent_adjacent_skill_is_still_surfaced():
    store, _, _ = store_with(2)
    (f,) = build_findings(OPP, None, None, None, _market(resume_gap=False), store, resume_supplied=False, today=TODAY)
    assert f.fact.endswith("It is not mentioned in this listing.")


def test_empty_market_yields_nothing():
    assert build_findings(OPP, None, None, None, MarketAnalysis(role_query="SWE"), EvidenceStore(CO), today=TODAY) == []


# ----------------------------------------------------------------- timing

def _timed(**kw):
    store = EvidenceStore(CO)
    item = store.add_opportunity(OPP.model_copy(update=kw), "The supplied listing")
    return store, item.id


def test_deadline_that_has_passed_is_reported_neutrally():
    store, lid = _timed(application_deadline="2026-09-20")
    (f,) = build_findings(OPP.model_copy(update={"application_deadline": "2026-09-20"}), None, None, None, None, store, listing_evidence_id=lid, today=TODAY)
    assert f.kind == "deadline" and "10 day(s) before this investigation (2026-09-30)" in f.fact
    assert "may no longer be accepting applications, or the date on the page may be outdated" in f.interpretation


def test_deadline_soon_far_and_unparseable():
    store, lid = _timed()
    def run(deadline):
        return build_findings(OPP.model_copy(update={"application_deadline": deadline}), None, None, None, None, store, listing_evidence_id=lid, today=TODAY)
    assert "5 day(s) after this investigation" in run("2026-10-05")[0].fact
    assert run("2027-01-31") == [] and run("03/04/2026") == [] and run(None) == []


def test_old_listing_is_flagged_and_recent_one_is_not():
    store, lid = _timed()
    old = build_findings(OPP.model_copy(update={"date_posted": "2026-06-23"}), None, None, None, None, store, listing_evidence_id=lid, today=TODAY)
    assert old[0].kind == "freshness" and "99 days before this investigation" in old[0].fact
    assert build_findings(OPP.model_copy(update={"date_posted": "2026-09-10"}), None, None, None, None, store, listing_evidence_id=lid, today=TODAY) == []


def test_timing_findings_need_the_listing_to_be_citable():
    opp = OPP.model_copy(update={"application_deadline": "2026-09-20"})
    assert build_findings(opp, None, None, None, None, EvidenceStore(CO), listing_evidence_id=None, today=TODAY) == []


# ---------------------------------------------------------------- language

def test_every_generated_finding_string_is_neutral():
    store, jobs, news = store_with(4, 1)
    opp = OPP.model_copy(update={"application_deadline": "2026-09-01", "date_posted": "2026-05-01"})
    listing = store.add_opportunity(opp, "The supplied listing").id
    cands = hiring([comp("experience", "Among 3 comparable listings ... vary.", jobs, 0.8), comp("skills", "Docker appears in 4 of 8 ...", jobs, 0.7)])
    for f in build_findings(opp, None, cands, None, _market(True), store, resume_supplied=True, listing_evidence_id=listing, today=TODAY):
        assert is_neutral(f"{f.title} {f.fact} {f.interpretation} {f.action or ''}"), f


def test_select_findings_is_pure_and_handles_empty():
    assert select_findings([]) == []
