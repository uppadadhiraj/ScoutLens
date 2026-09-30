"""Company Agent: establishes who the employer is, using only retrieved results.

Identity comes from Google's knowledge panel and organic results. The official website is the
listing's own domain, the panel's website, or (weakest) a top result whose domain matches the company
name; the basis is always recorded. Products and funding are *verbatim excerpts* with citations, not
paraphrases. The overview is the panel description, or LLM prose that survives citation validation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from models.company import CompanyFact, CompanyProfile
from models.opportunity import OpportunityProfile
from models.plan import PlannedSearch
from models.search import KnowledgeGraph, NewsItem, SearchHit
from services.evidence.citations import validate_prose
from services.evidence.store import EvidenceStore
from services.llm.base import LLMProvider
from services.serpapi.client import SerpApiClient
from services.serpapi.search import GoogleResults, fetch_google
from utils.domains import is_job_board_or_ats, origin_of, registered_domain
from utils.errors import ScoutLensError
from utils.language_guard import is_neutral
from utils.text import GENERIC_TRAILING_WORDS, mentions_company, normalize_company, truncate

logger = logging.getLogger(__name__)

FUNDING_PATTERN = re.compile(
    r"\b(raised|raises|raising|funding|funded|series [a-f]\b|seed round|pre-seed|investors?|valuation|"
    r"acquired|acquisition|acquires|ipo|backed by)\b",
    re.I,
)
_MAX_RELEVANT_HITS = 6
_MAX_PRODUCT_FACTS = 2
_MAX_FUNDING_FACTS = 3
_MAX_LLM_EXCERPTS = 5
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_OVERVIEW_SYSTEM = (
    "You write neutral, factual company overviews for job seekers using ONLY the numbered excerpts provided. "
    "Every sentence must end with one or more citation markers such as [E3] naming the excerpts that support it. "
    "Do not add facts that are not in the excerpts. Do not praise, criticise or evaluate the company. "
    "If the excerpts support nothing useful, reply with an empty string."
)


@dataclass
class CompanyResult:
    profile: CompanyProfile
    kg_founded_year: int | None = None
    has_knowledge_graph: bool = False
    relevant_hits: list[SearchHit] = field(default_factory=list)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _domain_matches_company(url: str, company: str) -> bool:
    """Whether the URL's registered-domain label *is* the company's name.

    Deliberately strict, because a match makes the domain "official" (Tier 1): the label must equal the
    name with hyphens removed, or the name minus one trailing generic word ("Razorpay Software" ->
    razorpay.com). Lookalikes such as ``northwindlabs-review.com`` never match.
    """
    label = _slug(registered_domain(url).split(".")[0])
    words = normalize_company(company).split()
    if not label or not words:
        return False
    accepted = {_slug("".join(words))}
    if len(words) > 1 and words[-1] in GENERIC_TRAILING_WORDS:
        accepted.add(_slug("".join(words[:-1])))
    return label in accepted


class CompanyAgent:
    def __init__(self, client: SerpApiClient, store: EvidenceStore, llm: LLMProvider | None = None) -> None:
        self._client = client
        self._store = store
        self._llm = llm

    # ------------------------------------------------------------------ public

    def investigate(
        self,
        opp: OpportunityProfile,
        identity: PlannedSearch,
        *,
        extra_identity: list[PlannedSearch] | None = None,
        refresh: bool = False,
    ) -> CompanyResult:
        """Run the identity search (and any extra identity-type searches) and assemble the profile."""
        company = opp.company_name or ""
        profile = CompanyProfile(name=company)
        results = [self._run(identity, refresh)]
        for step in extra_identity or []:
            results.append(self._run(step, refresh))

        kg = next((r.knowledge_graph for r in results if r.knowledge_graph), None)
        hits = self._relevant_hits(company, [h for r in results for h in r.hits])
        self._establish_website(profile, opp, kg, hits)
        kg_year = self._apply_knowledge_graph(profile, kg)
        self._add_product_facts(profile, hits)
        self._build_overview(profile, hits, kg)

        profile.identity_confirmed = bool(kg or profile.website)
        all_hits = [h for r in results for h in r.hits]
        profile.ambiguous = not kg and bool(all_hits) and len(hits) / len(all_hits) < 0.3
        if profile.ambiguous:
            profile.gaps.append(
                "Search results did not clearly identify this company (few results mention its name); "
                "findings about it should be verified."
            )
        if not profile.website:
            profile.gaps.append("Insufficient evidence: the official website could not be identified.")
        if not profile.overview:
            profile.gaps.append("Insufficient evidence: no company overview was found in the retrieved results.")
        return CompanyResult(profile=profile, kg_founded_year=kg_year, has_knowledge_graph=kg is not None, relevant_hits=hits)

    def retry_website(self, result: CompanyResult, opp: OpportunityProfile, step: PlannedSearch, *, refresh: bool = False) -> None:
        """A dedicated "official website" search for when the first search did not establish one."""
        profile = result.profile
        if profile.website:
            return
        found = self._run(step, refresh)
        hits = self._relevant_hits(profile.name, found.hits)
        result.relevant_hits = [*result.relevant_hits, *(h for h in hits if h not in result.relevant_hits)]
        self._establish_website(profile, opp, found.knowledge_graph, hits)
        if profile.website:
            profile.identity_confirmed = True
            profile.gaps = [g for g in profile.gaps if "official website" not in g]

    def add_funding(self, result: CompanyResult, step: PlannedSearch, *, refresh: bool = False) -> None:
        """Run the funding search and attach verbatim funding excerpts (with citations) to the profile."""
        profile = result.profile
        found = self._run(step, refresh)
        candidates = [h for h in self._relevant_hits(profile.name, found.hits) if FUNDING_PATTERN.search(f"{h.title} {h.snippet}")]
        for hit in candidates[:_MAX_FUNDING_FACTS]:
            item = self._store.add_hit(hit, f"Search result mentions funding or investment for {profile.name}")
            profile.funding.append(CompanyFact(text=truncate(hit.snippet or hit.title, 300), evidence_ids=[item.id]))
        if not profile.funding:
            profile.gaps.append("No funding information was found in the retrieved results.")

    def add_funding_from_news(self, profile: CompanyProfile, news: list[NewsItem]) -> None:
        """News headlines/snippets that mention funding also count, with the news item as the citation."""
        seen = {f.text for f in profile.funding}
        for item in news:
            text = truncate(item.snippet or item.title, 300)
            if len(profile.funding) >= _MAX_FUNDING_FACTS or text in seen:
                continue
            if mentions_company(f"{item.title} {item.snippet}", profile.name) and FUNDING_PATTERN.search(f"{item.title} {item.snippet}"):
                evidence = self._store.add_news(item, f"News coverage mentions funding or investment for {profile.name}")
                profile.funding.append(CompanyFact(text=text, evidence_ids=[evidence.id]))
                seen.add(text)
        if profile.funding:
            profile.gaps = [g for g in profile.gaps if not g.startswith("No funding information")]

    # ---------------------------------------------------------------- internals

    def _run(self, step: PlannedSearch, refresh: bool) -> GoogleResults:
        try:
            return fetch_google(self._client, step.query, refresh=refresh)
        finally:
            self._store.register_searches(self._client.records)

    def _relevant_hits(self, company: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Organic results that actually concern the company (by name in title/snippet or by domain)."""
        relevant = [
            h for h in hits
            if mentions_company(h.title, company) or mentions_company(h.snippet, company) or _domain_matches_company(h.url, company)
        ]
        return relevant[:_MAX_RELEVANT_HITS]

    def _establish_website(
        self, profile: CompanyProfile, opp: OpportunityProfile, kg: KnowledgeGraph | None, hits: list[SearchHit]
    ) -> None:
        listing_url = opp.application_url or opp.source_url
        website, basis = None, None
        if kg and kg.website:
            website, basis = kg.website, "Listed as the website in Google's knowledge panel"
        elif listing_url and not is_job_board_or_ats(listing_url) and _domain_matches_company(listing_url, opp.company_name or ""):
            website, basis = origin_of(listing_url), "The listing itself is hosted on this domain"
        else:
            match = next(
                (h for h in hits[:5] if not is_job_board_or_ats(h.url) and _domain_matches_company(h.url, profile.name)), None
            )
            if match:
                website, basis = origin_of(match.url), "Inferred: top search result whose domain matches the company name"
        if not website:
            return
        self._store.register_official_domain(website)
        if listing_url and not is_job_board_or_ats(listing_url) and _domain_matches_company(listing_url, opp.company_name or ""):
            self._store.register_official_domain(listing_url)
        profile.website, profile.website_basis = website, basis
        if kg and kg.website == website:
            item = self._store.add_knowledge_graph(
                kg, f"Official website: {website}", text=f"Google's knowledge panel for {kg.title} lists the website {website}.", url=website
            )
            if item:
                profile.website_evidence_ids.append(item.id)
        else:
            hit = next((h for h in hits if registered_domain(h.url) == registered_domain(website)), None)
            if hit:
                profile.website_evidence_ids.append(self._store.add_hit(hit, f"Official website: {website}").id)

    def _apply_knowledge_graph(self, profile: CompanyProfile, kg: KnowledgeGraph | None) -> int | None:
        if not kg:
            return None
        profile.industry = kg.type
        if kg.attributes:
            listed = "; ".join(f"{k}: {v}" for k, v in kg.attributes.items())
            item = self._store.add_knowledge_graph(kg, f"Facts listed in Google's knowledge panel for {kg.title}", text=listed)
            if item:
                for key, value in kg.attributes.items():
                    profile.attributes[key] = CompanyFact(text=value, evidence_ids=[item.id])
        founded = kg.attributes.get("founded", "")
        year = _YEAR.search(founded)
        return int(year.group(0)) if year else None

    def _add_product_facts(self, profile: CompanyProfile, hits: list[SearchHit]) -> None:
        """Snippets describing the company: official-site excerpts first, otherwise other relevant results."""
        official = [h for h in hits if registered_domain(h.url) in self._store.company_domains and h.snippet]
        pool = official or [h for h in hits if h.snippet]
        for hit in pool[:_MAX_PRODUCT_FACTS]:
            item = self._store.add_hit(hit, f"Description of {profile.name}'s business found in search results")
            profile.products.append(CompanyFact(text=truncate(hit.snippet, 300), evidence_ids=[item.id]))

    def _build_overview(self, profile: CompanyProfile, hits: list[SearchHit], kg: KnowledgeGraph | None) -> None:
        if kg and kg.description:
            item = self._store.add_knowledge_graph(kg, f"Overview of {kg.title}", text=kg.description)
            if item:
                profile.overview, profile.overview_source = kg.description, "knowledge_graph"
                profile.overview_evidence_ids = [item.id]
                return
        if self._llm and profile.products:
            if self._llm_overview(profile):
                return
        if profile.products:
            fact = profile.products[0]
            profile.overview, profile.overview_source = fact.text, "search_snippet"
            profile.overview_evidence_ids = list(fact.evidence_ids)

    def _llm_overview(self, profile: CompanyProfile) -> bool:
        excerpts = [(eid, self._store.get(eid)) for fact in profile.products for eid in fact.evidence_ids]
        lines = [f"[{eid}] {item.evidence_text}" for eid, item in excerpts[:_MAX_LLM_EXCERPTS] if item]
        if not lines:
            return False
        try:
            text = self._llm.complete(_OVERVIEW_SYSTEM, f"Company: {profile.name}\nExcerpts:\n" + "\n".join(lines) + "\n\nWrite 2-3 sentences.")  # type: ignore[union-attr]
        except ScoutLensError as exc:
            logger.info("LLM overview skipped: %s", exc)
            return False
        checked = validate_prose(text, self._store)
        if checked.is_empty or not is_neutral(checked.text):
            return False
        profile.overview, profile.overview_source = checked.text, "llm"
        profile.overview_evidence_ids = checked.evidence_ids
        return True
