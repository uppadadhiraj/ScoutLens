"""Google Jobs via SerpApi: related current listings, the raw material for hiring and market signals."""
from __future__ import annotations

import logging
import re
from typing import Any

from models.search import ApplyLink, JobListing, SearchQuery
from services.serpapi.client import SerpApiClient
from services.serpapi.parsing import as_dict, as_list, build_all, clean_text
from utils.errors import SerpApiError

logger = logging.getLogger(__name__)


def _strings(items: Any) -> list[str]:
    return [t for raw in as_list(items) if (t := clean_text(raw))]


def _highlights(job: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Split ``job_highlights`` into (qualifications, responsibilities)."""
    qualifications: list[str] = []
    responsibilities: list[str] = []
    for block in as_list(job.get("job_highlights")):
        block = as_dict(block)
        heading = (clean_text(block.get("title")) or "").lower()
        if "qualif" in heading or "requirement" in heading or "skill" in heading:
            qualifications += _strings(block.get("items"))
        elif "responsib" in heading:
            responsibilities += _strings(block.get("items"))
    return qualifications, responsibilities


def _apply_links(job: dict[str, Any]) -> list[ApplyLink]:
    links: list[ApplyLink] = []
    for option in as_list(job.get("apply_options")):
        option = as_dict(option)
        title, link = clean_text(option.get("title")), clean_text(option.get("link"))
        if title and link:
            try:
                links.append(ApplyLink(title=title, url=link))
            except ValueError:
                continue
    return links


def _optional_url(value: Any) -> str | None:
    url = clean_text(value)
    return url if url and url.startswith(("http://", "https://")) else None


def _job(search_id: str, job: dict[str, Any]) -> JobListing:
    detected = as_dict(job.get("detected_extensions"))
    qualifications, responsibilities = _highlights(job)
    via = clean_text(job.get("via"))
    wfh = detected.get("work_from_home")
    return JobListing(
        search_id=search_id,
        job_id=clean_text(job.get("job_id")),
        title=clean_text(job.get("title")) or "",
        company_name=clean_text(job.get("company_name")),
        location=clean_text(job.get("location")),
        via=re.sub(r"(?i)^via\s+", "", via) if via else None,
        description=clean_text(job.get("description")) or "",
        posted_at=clean_text(detected.get("posted_at")),
        schedule_type=clean_text(detected.get("schedule_type")),
        salary=clean_text(detected.get("salary")),
        work_from_home=wfh if isinstance(wfh, bool) else None,
        qualifications=qualifications,
        responsibilities=responsibilities,
        apply_links=_apply_links(job),
        share_link=_optional_url(job.get("share_link")),
    )


def parse_jobs(data: dict[str, Any], search_id: str) -> list[JobListing]:
    listings = build_all(as_list(data.get("jobs_results")), lambda job: _job(search_id, job))
    return [job for job in listings if job.title]


def next_page_token(data: dict[str, Any]) -> str | None:
    return clean_text(as_dict(data.get("serpapi_pagination")).get("next_page_token"))


def dedupe_jobs(listings: list[JobListing]) -> list[JobListing]:
    """Drop repeats (same posting surfaced by several queries/pages), keeping first occurrence."""
    seen: set[str] = set()
    unique: list[JobListing] = []
    for job in listings:
        key = job.job_id or "|".join(
            (v or "").lower() for v in (job.title, job.company_name, job.location)
        )
        if key not in seen:
            seen.add(key)
            unique.append(job)
    return unique


def fetch_jobs(
    client: SerpApiClient, query: SearchQuery, *, refresh: bool = False
) -> list[JobListing]:
    """Run a Google Jobs query, following pagination up to ``query.pages`` (1 credit per page).

    A failure on the first page is raised; a failure on a later page just ends pagination so the
    listings already collected are not thrown away.
    """
    listings: list[JobListing] = []
    page_query = query
    for page in range(query.pages):
        try:
            response = client.search(page_query, refresh=refresh)
        except SerpApiError:
            if page == 0:
                raise
            logger.info("Job pagination stopped after page %d", page)
            break
        listings += parse_jobs(response.data, response.search_id)
        token = next_page_token(response.data)
        if not token:
            break
        page_query = query.model_copy(update={"params": {**query.params, "next_page_token": token}})
    return dedupe_jobs(listings)
