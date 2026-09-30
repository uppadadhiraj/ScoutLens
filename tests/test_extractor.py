from pathlib import Path

import pytest

from services.scraping.extractor import (
    extract_from_html,
    extract_from_text,
    infer_company_from_url,
    split_title,
)
from services.scraping.html_text import fragment_to_text, html_to_text
from utils.errors import ExtractionError

FIXTURES = Path(__file__).parent / "fixtures"


def _html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ------------------------------------------------------------------- JSON-LD

def test_jsonld_posting_is_fully_extracted():
    opp = extract_from_html(_html("job_jsonld.html"), "https://www.northwindlabs.example/careers/swe-intern")
    assert opp.extraction_method == "jsonld"
    assert opp.company_name == "Northwind Labs"
    assert opp.role_title == "Software Engineer Intern"
    assert opp.job_type == "Internship"
    assert opp.location == "Bengaluru, Karnataka, IN"
    assert opp.salary == "INR 25,000–40,000 per month"
    assert opp.application_deadline == "2026-10-31"
    assert opp.date_posted == "2026-09-10"
    assert opp.education_required == "bachelor degree"
    assert opp.application_url == "https://www.northwindlabs.example/careers/swe-intern"
    assert opp.field_sources["company_name"] == "jsonld"
    assert opp.is_investigable


def test_jsonld_zero_months_experience_means_no_experience_required():
    opp = extract_from_html(_html("job_jsonld.html"), "https://www.northwindlabs.example/x")
    assert (opp.experience_min_years, opp.experience_max_years) == (0.0, None)
    assert opp.experience_required == "0 months"


def test_jsonld_description_sections_drive_skill_levels():
    opp = extract_from_html(_html("job_jsonld.html"), "https://www.northwindlabs.example/x")
    assert opp.responsibilities == ["Build internal tools in Python", "Write SQL queries for reports"]
    assert opp.qualifications[0] == "Pursuing a B.Tech in Computer Science"
    assert opp.required_skills == ["Python", "Git", "REST APIs", "SQL"]
    assert opp.preferred_skills == ["Docker", "AWS"]
    # "Kubernetes" only appears under Benefits, so it must not be attributed to the role.
    assert "Kubernetes" not in opp.all_skills


# ------------------------------------------------------------ plain HTML page

def test_plain_page_falls_back_to_meta_and_text():
    opp = extract_from_html(_html("job_plain.html"), "https://contoso.example/careers/analyst")
    assert opp.extraction_method == "html"
    assert opp.company_name == "Contoso Analytics"
    assert opp.role_title == "Data Analyst Intern"
    assert opp.field_sources["company_name"] == "meta"
    assert opp.job_type == "Internship"
    assert opp.location == "Pune, India (Hybrid)"
    assert opp.remote_or_onsite == "hybrid"
    assert opp.application_deadline == "15 October 2026"


def test_plain_page_salary_ignores_funding_figure_and_freshers_are_detected():
    opp = extract_from_html(_html("job_plain.html"), "https://contoso.example/x")
    assert opp.salary == "₹15,000 - ₹20,000 per month"
    assert opp.experience_min_years == 0.0 and opp.experience_max_years is None
    assert "fresher" in opp.experience_required.lower()
    assert "Bachelor" in opp.education_required


def test_plain_page_skills_are_split_and_ordinary_english_is_not_a_skill():
    opp = extract_from_html(_html("job_plain.html"), "https://contoso.example/x")
    assert {"SQL", "Excel", "Python", "Pandas", "Power BI"} <= set(opp.required_skills)
    assert opp.preferred_skills == ["Tableau"]  # "...people who excel at teamwork" is not Excel
    assert "Excel" not in opp.preferred_skills


def test_page_chrome_is_excluded_from_description():
    opp = extract_from_html(_html("job_plain.html"), "https://contoso.example/x")
    assert "Privacy | Terms" not in opp.description


def test_js_only_page_asks_for_manual_paste():
    with pytest.raises(ExtractionError) as info:
        extract_from_html(_html("job_js_only.html"), "https://spa.example/jobs/1")
    assert "paste" in info.value.user_message.lower()


def test_blocked_and_login_pages_are_reported_as_such():
    with pytest.raises(ExtractionError, match="blocked"):
        extract_from_html("<html><body><h1>Access denied</h1><p>Verify you are human</p></body></html>", "https://x.example/")
    with pytest.raises(ExtractionError, match="signing in"):
        extract_from_html("<html><body><p>Sign in to view this job</p></body></html>", "https://x.example/")


def test_malformed_jsonld_is_ignored_not_fatal():
    html = (
        "<html><head><title>Backend Engineer - Acme | Careers</title>"
        '<script type="application/ld+json">{ not json </script></head><body><main>'
        + "<p>We are hiring a backend engineer to build services in Python. </p>" * 8
        + "</main></body></html>"
    )
    opp = extract_from_html(html, "https://acme.example/jobs/1")
    assert opp.extraction_method == "html"
    assert opp.company_name == "Acme" and opp.role_title == "Backend Engineer"


def test_trailing_comma_jsonld_is_tolerated():
    html = (
        '<html><head><script type="application/ld+json">'
        '{"@type": "JobPosting", "title": "QA Engineer", "hiringOrganization": {"name": "Acme",},'
        ' "description": "Short."}</script></head><body></body></html>'
    )
    opp = extract_from_html(html, "https://acme.example/jobs/2")
    assert (opp.company_name, opp.role_title) == ("Acme", "QA Engineer")


def test_jsonld_inside_graph_is_found():
    html = (
        '<html><head><script type="application/ld+json">'
        '{"@graph": [{"@type": "WebSite"}, {"@type": "JobPosting", "title": "SRE", '
        '"hiringOrganization": "Globex", "description": "Run things."}]}</script></head><body></body></html>'
    )
    opp = extract_from_html(html, "https://globex.example/jobs/2")
    assert (opp.company_name, opp.role_title) == ("Globex", "SRE")


# ------------------------------------------------------------ manual paste

PASTED = """Data Engineer Intern
Location: Remote
Responsibilities:
- Build ETL pipelines in Python
- Maintain Airflow DAGs
Requirements:
- 0-1 years of experience
- SQL and Docker
Nice to have:
- Kafka
Apply by 30 November 2026.
"""


def test_manual_text_uses_user_supplied_company_and_flags_guessed_title():
    opp = extract_from_text(PASTED, company_name="Globex Corp")
    assert opp.extraction_method == "manual"
    assert opp.company_name == "Globex Corp" and opp.field_sources["company_name"] == "manual"
    assert opp.role_title == "Data Engineer Intern"
    assert any("first line" in n for n in opp.notes)
    assert opp.remote_or_onsite == "remote"
    assert (opp.experience_min_years, opp.experience_max_years) == (0.0, 1.0)
    # explicit requirements first (SQL, Docker), then skills named only in responsibilities
    assert opp.required_skills == ["SQL", "Docker", "ETL", "Python", "Airflow"]
    assert opp.preferred_skills == ["Apache Kafka"]
    assert opp.application_deadline == "30 November 2026"


def test_manual_text_without_company_is_not_investigable():
    opp = extract_from_text(PASTED)
    assert opp.company_name is None and not opp.is_investigable


def test_manual_text_user_role_overrides_guess():
    opp = extract_from_text(PASTED, company_name="Globex", role_title="Platform Intern")
    assert opp.role_title == "Platform Intern" and opp.field_sources["role_title"] == "manual"


def test_manual_text_too_short_is_rejected():
    with pytest.raises(ExtractionError):
        extract_from_text("hi")


def test_unlabelled_listing_flags_that_requirement_level_is_unknown():
    text = "We need someone who knows Python and Docker for our platform team in the city. " * 3
    opp = extract_from_text(text, company_name="Acme", role_title="Engineer")
    assert opp.required_skills == ["Python", "Docker"]
    assert any("requirement level" in n for n in opp.notes)


# Shape seen on real ATS pages (e.g. Lever): plain-text JSON-LD description, headings are bare lines
# with no markup or colon, and newlines are sometimes double-escaped to a literal backslash-n.
BARE_HEADING_TEXT = (
    "About us\nWe build logistics software for retailers across many countries and time zones.\n"
    "\\n\nResponsibilities\n- Design and maintain REST APIs in Python\n- Own deployments on AWS\n"
    "Requirements\n- 4+ years of experience with backend systems\n- Experience with Docker\n"
    "Nice to have\n- Kubernetes\nBenefits\n- Learning budget for Terraform courses\n"
)


def test_bare_line_headings_and_escaped_newlines_in_plain_text_jsonld():
    import json

    html = (
        '<html><head><script type="application/ld+json">'
        + json.dumps({"@type": "JobPosting", "title": "Backend Engineer", "hiringOrganization": {"name": "Acme"},
                      "description": BARE_HEADING_TEXT})
        + "</script></head><body></body></html>"
    )
    opp = extract_from_html(html, "https://acme.example/jobs/9")
    assert opp.responsibilities == ["Design and maintain REST APIs in Python", "Own deployments on AWS"]
    assert opp.qualifications == ["4+ years of experience with backend systems", "Experience with Docker"]
    # requirement lines first (Docker), then responsibilities in text order
    assert opp.required_skills == ["Docker", "REST APIs", "Python", "AWS"]
    assert opp.preferred_skills == ["Kubernetes"]
    assert "Terraform" not in opp.all_skills  # benefits text is not a requirement
    assert "\\n" not in opp.description
    assert opp.experience_min_years == 4.0


def test_short_requirement_like_lines_are_items_not_headings():
    from services.scraping.text_parsers import split_sections

    sections = split_sections("Requirements\nExperience with Docker\nSkills and experience\nPython")
    assert sections["requirements"] == ["Experience with Docker", "Python"]


# ------------------------------------------------------- title / URL inference

@pytest.mark.parametrize(
    "title, expected",
    [
        ("Backend Engineer - Northwind Labs | Careers", ("Backend Engineer", "Northwind Labs")),
        ("Northwind Labs - Backend Engineer", ("Backend Engineer", "Northwind Labs")),
        ("Software Engineer Intern at Northwind Labs | Careers", ("Software Engineer Intern", "Northwind Labs")),
        ("Careers", (None, None)),
        ("Northwind Labs", (None, "Northwind Labs")),
        ("Senior Data Analyst", ("Senior Data Analyst", None)),
        ("", (None, None)),
    ],
)
def test_split_title(title, expected):
    assert split_title(title) == expected


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://boards.greenhouse.io/northwind-labs/jobs/123", "Northwind Labs"),
        ("https://jobs.lever.co/acme/abc-123", "Acme"),
        ("https://acme.bamboohr.com/careers/5", "Acme"),
        ("https://careers.northwindlabs.com/jobs/1", "Northwindlabs"),
        ("https://www.contoso.co.in/careers", "Contoso"),
        ("https://www.linkedin.com/jobs/view/123", None),
        ("https://in.indeed.com/viewjob?jk=1", None),
    ],
)
def test_infer_company_from_url(url, expected):
    assert infer_company_from_url(url) == expected


def test_company_inferred_from_url_carries_a_verify_note():
    html = "<html><head><title>Backend Engineer</title></head><body><main>" + "<p>Build APIs in Python for us daily.</p>" * 12 + "</main></body></html>"
    opp = extract_from_html(html, "https://jobs.lever.co/acme/abc")
    assert opp.company_name == "Acme" and opp.field_sources["company_name"] == "url"
    assert any("verify" in n.lower() for n in opp.notes)


# --------------------------------------------------------------- html → text

def test_html_to_text_marks_headings_bullets_and_pseudo_headings():
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(
        "<div><h2>Role</h2><p>Intro <b>bold</b> text</p><p><strong>Requirements</strong></p>"
        "<ul><li>One</li><li>Two <ul><li>Nested</li></ul></li></ul>a<br>b</div>",
        "lxml",
    )
    lines = html_to_text(soup).splitlines()
    assert lines[:4] == ["## Role", "Intro bold text", "## Requirements", "- One"]
    assert "- Nested" in lines and lines[-2:] == ["a", "b"]


def test_fragment_to_text_handles_escaped_html_and_plain_text():
    assert fragment_to_text("&lt;li&gt;Python&lt;/li&gt;") == "- Python"
    assert fragment_to_text("line one\n\n  line two ") == "line one\nline two"
