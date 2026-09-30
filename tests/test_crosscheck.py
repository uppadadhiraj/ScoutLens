import pytest

from models.company import CompanyFact, CompanyProfile
from models.opportunity import OpportunityProfile
from models.report import HiringSignal, SignalComparison
from models.search import JobListing, KnowledgeGraph, NewsItem, SearchHit
from services.evidence.crosscheck import run_crosschecks
from services.evidence.store import EvidenceStore

CO = "Northwind Labs"
SITE = "https://www.northwindlabs.example/"
OPP = OpportunityProfile(company_name=CO, role_title="Software Engineer Intern", source_url="https://www.northwindlabs.example/careers/x")


def by_claim(checks, fragment):
    return next(c for c in checks if fragment in c.claim)


# ----------------------------------------------------------- store additions

def test_opportunity_evidence_is_the_fetched_page_and_tiered_by_domain():
    store = EvidenceStore(CO, ["northwindlabs.example"])
    item = store.add_opportunity(OPP)
    assert (item.source_url, item.search_id, item.tier, item.source_type) == (OPP.source_url, "page", 1, "job")
    assert item.retrieved_at == OPP.extracted_at
    aggregator = OPP.model_copy(update={"source_url": "https://www.linkedin.com/jobs/view/1"})
    assert EvidenceStore(CO).add_opportunity(aggregator).tier == 3


def test_pasted_text_cannot_be_cited():
    assert EvidenceStore(CO).add_opportunity(OPP.model_copy(update={"extraction_method": "manual"})) is None
    assert EvidenceStore(CO).add_opportunity(OPP.model_copy(update={"source_url": None})) is None


def test_confidence_upgrades_one_level_and_never_above_high():
    store = EvidenceStore(CO)
    low = store.add_hit(SearchHit(search_id="s", title="Other", url="https://x.example/a", snippet="nothing"), "c")
    high = store.add_hit(SearchHit(search_id="s", title="Northwind Labs", url="https://y.example/b", snippet="Northwind Labs"), "c")
    store.register_official_domain("https://y.example/")
    assert (low.confidence, high.confidence) == ("low", "high")
    store.upgrade_confidence([low.id, high.id, "E404"])
    assert (low.confidence, high.confidence) == ("medium", "high")
    store.upgrade_confidence([low.id])
    assert low.confidence == "high"


# ------------------------------------------------------------------- website

def _company_with_site(basis, store, *, with_hit=True):
    kg = KnowledgeGraph(search_id="s1", title=CO, description="d", website=SITE.rstrip("/"))
    store.register_official_domain(SITE)
    ids = [store.add_knowledge_graph(kg, f"Official website: {SITE}", url=SITE).id]
    if with_hit:
        store.add_hit(SearchHit(search_id="s1", title="Northwind Labs", url=SITE, snippet="Northwind Labs builds software."), "Description")
    return CompanyProfile(name=CO, website=SITE, website_basis=basis, website_evidence_ids=ids)


def test_website_is_corroborated_by_multiple_independent_signals():
    store = EvidenceStore(CO)
    company = _company_with_site("Listed as the website in Google's knowledge panel", store)
    check = by_claim(run_crosschecks(OPP, company, None, store), "Official website")
    assert check.status == "corroborated"
    assert "knowledge panel lists it" in check.detail and "hosted on it" in check.detail and "search result" in check.detail
    assert store.get(company.website_evidence_ids[0]).confidence == "high"


def test_website_from_a_single_signal_is_reported_as_single_source():
    store = EvidenceStore(CO)
    company = _company_with_site("Inferred: top search result whose domain matches the company name", store, with_hit=False)
    opp = OPP.model_copy(update={"source_url": "https://www.linkedin.com/jobs/view/1"})
    check = by_claim(run_crosschecks(opp, company, None, store), "Official website")
    assert check.status == "single_source"


def test_no_website_means_no_website_check():
    assert run_crosschecks(OPP, CompanyProfile(name=CO), None, EvidenceStore(CO)) == []


# ------------------------------------------------------------------- funding

def _funding_company(store, urls):
    facts = []
    for i, url in enumerate(urls):
        item = store.add_news(NewsItem(search_id="s3", title=f"Northwind raises {i}", url=url, publisher="Reuters"), f"funding {i}")
        facts.append(CompanyFact(text=f"raised {i}", evidence_ids=[item.id]))
    return CompanyProfile(name=CO, funding=facts)


def test_funding_reported_by_two_domains_is_corroborated():
    store = EvidenceStore(CO)
    company = _funding_company(store, ["https://a.example/1", "https://www.b.example/2"])
    check = by_claim(run_crosschecks(OPP, company, None, store), "Funding")
    assert check.status == "corroborated" and "2 independent sources" in check.detail


def test_funding_from_one_domain_counts_as_one_source_even_with_two_articles():
    store = EvidenceStore(CO)
    company = _funding_company(store, ["https://a.example/1", "https://a.example/2"])
    check = by_claim(run_crosschecks(OPP, company, None, store), "Funding")
    assert check.status == "single_source" and "unverified information" in check.detail


def test_no_funding_no_check():
    assert run_crosschecks(OPP, CompanyProfile(name=CO), None, EvidenceStore(CO)) == []


# ------------------------------------------------------------------- listing

def _hiring(found: bool, store):
    comparisons = []
    if found:
        job = JobListing(search_id="s2", title=OPP.role_title, company_name=CO, via="LinkedIn", share_link="https://g.example/orig")
        item = store.add_job(job, "The supplied listing also appears in Google Jobs")
        comparisons.append(SignalComparison(kind="listing", fact="This listing also appears in Google Jobs (via LinkedIn).", evidence_ids=[item.id]))
    return HiringSignal(company=CO, sample_size=3, original_found=found, comparisons=comparisons)


def test_listing_seen_on_the_page_and_in_google_jobs_is_corroborated():
    store = EvidenceStore(CO)
    lid = store.add_opportunity(OPP).id
    check = by_claim(run_crosschecks(OPP, None, _hiring(True, store), store, listing_evidence_id=lid), "publicly posted")
    assert check.status == "corroborated" and lid in check.evidence_ids


def test_listing_not_found_in_google_jobs_is_neutral_about_it():
    store = EvidenceStore(CO)
    lid = store.add_opportunity(OPP).id
    check = by_claim(run_crosschecks(OPP, None, _hiring(False, store), store, listing_evidence_id=lid), "publicly posted")
    assert check.status == "single_source" and "may only reflect search coverage" in check.detail


def test_pasted_listing_cannot_be_verified_directly():
    store = EvidenceStore(CO)
    manual = OPP.model_copy(update={"extraction_method": "manual"})
    check = by_claim(run_crosschecks(manual, None, _hiring(False, store), store), "publicly posted")
    assert check.status == "insufficient" and "pasted manually" in check.detail


# ---------------------------------------------------------------- experience

@pytest.mark.parametrize(
    "fact, status, prefix",
    [
        ("All 3 comparable listings that state experience ask for the same as this one (0–2 years).", "corroborated", "All 3"),
        ("This listing states 0–2 years. Among 3 comparable listings ..., 2 state something different: 1–3 years (2 listings).", "conflicting", "Conflicting information: This listing"),
    ],
)
def test_experience_consistency_maps_to_neutral_statuses(fact, status, prefix):
    hiring = HiringSignal(company=CO, sample_size=3, comparisons=[SignalComparison(kind="experience", fact=fact, evidence_ids=[])])
    check = by_claim(run_crosschecks(OPP, None, hiring, EvidenceStore(CO)), "Experience requirement")
    assert check.status == status and check.detail.startswith(prefix)
