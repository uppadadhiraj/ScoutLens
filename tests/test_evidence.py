import pytest

from models.search import JobListing, KnowledgeGraph, NewsItem, SearchHit, SearchRecord
from services.evidence.citations import cited_ids, strip_citations, validate_prose
from services.evidence.source_quality import assess_confidence, classify_source
from services.evidence.store import EvidenceStore

COMPANY = "Northwind Labs"
OFFICIAL = frozenset({"northwindlabs.example"})


# ------------------------------------------------------------- source quality

@pytest.mark.parametrize(
    "url, kind, publisher, tier, label_part",
    [
        ("https://www.northwindlabs.example/careers", "web", None, 1, "Official company"),
        ("https://careers.northwindlabs.example/x", "job", None, 1, "Official company"),
        ("https://jobs.lever.co/northwind-labs/abc", "job", "Lever", 1, "employer's ATS"),
        ("https://boards.greenhouse.io/otherco/jobs/1", "job", None, 3, "Other web"),  # ATS of a different employer
        ("https://data.gov.in/x", "web", None, 1, "Government"),
        ("https://www.iitb.ac.in/x", "web", None, 1, "University"),
        ("https://arxiv.org/abs/1234", "web", None, 1, "Primary research"),
        ("https://www.reuters.com/business/x", "news", "Reuters", 2, "news"),
        ("https://obscure.example/story", "news", "The Economic Times", 2, "news"),
        ("https://www.linkedin.com/jobs/view/1", "job", "LinkedIn", 3, "aggregator"),
        ("https://in.indeed.com/viewjob?jk=1", "job", "Indeed", 3, "aggregator"),
        ("https://www.reddit.com/r/x", "web", None, 3, "Forum"),
        ("https://en.wikipedia.org/wiki/X", "knowledge_graph", "Wikipedia", 3, "community-edited"),
        ("https://random-blog.example/post", "web", None, 3, "unclassified"),
        ("https://random-blog.example/post", "knowledge_graph", "Google", 2, "knowledge panel"),
    ],
)
def test_classify_source(url, kind, publisher, tier, label_part):
    quality = classify_source(url, publisher=publisher, kind=kind, company_domains=OFFICIAL, company_name=COMPANY)
    assert quality.tier == tier
    assert label_part.lower() in quality.label.lower()


def test_lookalike_domain_is_not_official():
    quality = classify_source("https://northwindlabs.example.evil.com/", company_domains=OFFICIAL, company_name=COMPANY)
    assert quality.tier == 3


@pytest.mark.parametrize(
    "tier, relevance, expected",
    [(1, "high", "high"), (1, "medium", "medium"), (2, "high", "medium"), (3, "high", "low"), (1, "low", "low")],
)
def test_confidence(tier, relevance, expected):
    assert assess_confidence(tier, relevance) == expected


# ---------------------------------------------------------------------- store

def _hit(**kw):
    base = dict(search_id="s1", title="Northwind Labs | Logistics software", url="https://www.northwindlabs.example/",
                snippet="Northwind Labs helps retailers plan routes.")
    return SearchHit(**{**base, **kw})


def test_hit_evidence_copies_source_verbatim_and_is_tiered():
    store = EvidenceStore(COMPANY, OFFICIAL)
    item = store.add_hit(_hit(), "Company description")
    assert item.id == "E1"
    assert item.source_url == "https://www.northwindlabs.example/"
    assert item.evidence_text == "Northwind Labs helps retailers plan routes."
    assert (item.tier, item.source_type, item.relevance, item.confidence) == (1, "company_site", "high", "high")


def test_relevance_from_company_mention():
    store = EvidenceStore(COMPANY)
    assert store.add_hit(_hit(url="https://a.example/1"), "c").relevance == "high"
    assert store.add_hit(_hit(url="https://a.example/2", title="Logistics", snippet="by Northwind Labs"), "c").relevance == "medium"
    assert store.add_hit(_hit(url="https://a.example/3", title="Cats", snippet="Dogs"), "c").relevance == "low"


def test_duplicates_return_the_same_item_but_new_claims_get_new_ids():
    store = EvidenceStore(COMPANY)
    a = store.add_hit(_hit(), "claim one")
    assert store.add_hit(_hit(), "claim one") is a
    b = store.add_hit(_hit(), "claim two")
    assert b.id == "E2" and len(store) == 2


def test_registering_official_domain_retiers_existing_evidence():
    store = EvidenceStore(COMPANY)
    item = store.add_hit(_hit(), "c")
    assert item.tier == 3
    store.register_official_domain("https://www.northwindlabs.example/about")
    assert (item.tier, item.source_type, item.confidence) == (1, "company_site", "high")


def test_job_without_any_url_cannot_be_cited():
    store = EvidenceStore(COMPANY)
    assert store.add_job(JobListing(search_id="s1", title="Dev", company_name=COMPANY), "c") is None
    job = JobListing(search_id="s1", title="Dev", company_name=COMPANY, via="LinkedIn",
                     share_link="https://www.google.com/search?q=x", description="Python and SQL")
    item = store.add_job(job, "Current opening")
    assert item.source_type == "job" and item.publisher == "LinkedIn" and item.tier == 3


def test_news_evidence_uses_publisher_for_tier():
    store = EvidenceStore(COMPANY)
    news = NewsItem(search_id="s3", title="Northwind Labs raises Series B", url="https://x.example/a",
                    publisher="Reuters", date_text="09/29/2026, 08:07 AM, +0000 UTC", snippet="Funds will go to engineering.")
    item = store.add_news(news, "Recent coverage")
    assert (item.tier, item.date, item.relevance) == (2, "09/29/2026, 08:07 AM, +0000 UTC", "high")


def test_knowledge_graph_without_any_link_yields_no_evidence():
    store = EvidenceStore(COMPANY)
    kg = KnowledgeGraph(search_id="s1", title=COMPANY, description="d")
    assert store.add_knowledge_graph(kg, "overview") is None
    kg2 = kg.model_copy(update={"source_name": "Wikipedia", "source_url": "https://en.wikipedia.org/wiki/N"})
    item = store.add_knowledge_graph(kg2, "overview")
    assert item.tier == 3 and item.publisher == "Wikipedia"


def test_retrieved_at_comes_from_the_search_record():
    from datetime import datetime, timezone

    store = EvidenceStore(COMPANY)
    when = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    store.register_search(SearchRecord(id="s1", engine="google", query="q", fetched_at=when, from_cache=True))
    assert store.add_hit(_hit(), "c").retrieved_at == when


def test_existing_ids_filters_unknown():
    store = EvidenceStore(COMPANY)
    store.add_hit(_hit(), "c")
    assert store.existing_ids(["E1", "E9", "nonsense"]) == ["E1"]


# ------------------------------------------------------------------ citations

@pytest.fixture
def store():
    s = EvidenceStore(COMPANY)
    s.add_hit(_hit(url="https://a.example/1"), "a")
    s.add_hit(_hit(url="https://a.example/2"), "b")
    return s


def test_prose_keeps_only_sentences_with_valid_citations(store):
    text = "Northwind builds routing software [E1]. It has offices on Mars [E99]. It is widely loved. Founded in 2016 [E2]."
    result = validate_prose(text, store)
    assert result.text == "Northwind builds routing software [E1]. Founded in 2016 [E2]."
    assert result.evidence_ids == ["E1", "E2"] and result.dropped == 2


def test_prose_with_citation_after_the_full_stop_is_not_lost(store):
    result = validate_prose("Northwind builds routing software. [E1] Founded in 2016. [E2]", store)
    assert result.evidence_ids == ["E1", "E2"] and result.dropped == 0


def test_a_sentence_with_one_bad_citation_among_good_ones_is_dropped(store):
    assert validate_prose("Both facts [E1][E99].", store).is_empty


def test_empty_and_uncited_prose(store):
    assert validate_prose("", store).is_empty
    assert validate_prose("Completely uncited claim.", store).dropped == 1


def test_cited_ids_and_strip():
    assert cited_ids("A [E1] B [E2] C [E1]") == ["E1", "E2"]
    assert strip_citations("A [E1] B.") == "A B."
