import pytest

from services.scraping.text_parsers import (
    classify_heading,
    detect_job_type,
    detect_work_mode,
    format_experience,
    job_type_from_schema,
    parse_deadline,
    parse_education,
    parse_experience,
    parse_salary,
    split_sections,
)


@pytest.mark.parametrize(
    "text, lo, hi",
    [
        ("0-2 years of experience", 0, 2),
        ("1–3 yrs experience preferred", 1, 3),
        ("3+ years of relevant experience with Python", 3, None),
        ("Minimum 2 years experience in backend", 2, None),
        ("Experience: 4 years", 4, None),
    ],
)
def test_experience_numeric(text, lo, hi):
    info = parse_experience(text)
    assert info is not None
    assert (info.min_years, info.max_years) == (lo, hi)


def test_experience_range_beats_its_own_upper_bound():
    info = parse_experience("Requires 0-2 years of experience")
    assert info.raw.startswith("0-2 years")


def test_freshers_are_recognised_when_no_number_is_given():
    info = parse_experience("Freshers eligible to apply.")
    assert (info.min_years, info.max_years) == (0.0, None)


def test_years_that_are_not_experience_requirements_are_ignored():
    assert parse_experience("Founded 12 years ago, we serve retailers.") is None
    assert parse_experience("Our platform has 10 years of history.") is None


def test_format_experience():
    assert format_experience(0, 2) == "0–2 years"
    assert format_experience(3, None) == "3+ years"
    assert format_experience(0, None) == "freshers / no experience stated"
    assert format_experience(None, None) == "not stated"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Stipend: ₹15,000 - ₹20,000 per month", "₹15,000 - ₹20,000 per month"),
        ("Salary $80,000 to $100,000 per year", "$80,000 to $100,000 per year"),
        ("CTC: 12-18 LPA", "12-18 LPA"),
        ("This is an unpaid internship", "Unpaid"),
    ],
)
def test_salary(text, expected):
    assert parse_salary(text) == expected


def test_funding_and_revenue_figures_are_not_salaries():
    assert parse_salary("The company raised $40 million in funding last year.") is None
    assert parse_salary("Annual revenue of $2 million") is None
    assert parse_salary("No compensation details are given here.") is None


def test_education_picks_first_degree_line():
    lines = ["Great team", "Pursuing a B.Tech in Computer Science", "Bachelor's degree preferred"]
    assert parse_education(lines) == "Pursuing a B.Tech in Computer Science"
    assert parse_education(["We value people who can be flexible"]) is None  # "BE" must not match "be"


def test_deadline():
    assert parse_deadline("Last date to apply: 15 October 2026.") == "15 October 2026"
    assert parse_deadline("Apply by Oct 31, 2026") == "Oct 31, 2026"
    assert parse_deadline("Application deadline - 2026-11-05") == "2026-11-05"
    assert parse_deadline("Apply whenever you like") is None


def test_job_type():
    assert detect_job_type("Software Engineer Intern", "") == "Internship"
    assert detect_job_type("Engineer", "This is a full-time role") == "Full-time"
    assert detect_job_type("Engineer", "") is None
    assert job_type_from_schema("FULL_TIME") == "Full-time"
    assert job_type_from_schema("nonsense") is None


def test_work_mode():
    assert detect_work_mode("Engineer", "Remote", "") == "remote"
    assert detect_work_mode("Engineer", "Pune (Hybrid)", "") == "hybrid"
    assert detect_work_mode("Engineer", "Pune", "This is a fully remote role.") == "remote"
    assert detect_work_mode("Engineer", "Pune", "Work from office, 5 days") == "onsite"
    assert detect_work_mode("Engineer", "Pune", "We like remote controls") is None


@pytest.mark.parametrize(
    "heading, kind",
    [
        ("Responsibilities", "responsibilities"),
        ("What you'll do", "responsibilities"),
        ("Requirements", "requirements"),
        ("Preferred qualifications", "preferred"),  # preferred wins over "qualifications"
        ("Nice to have", "preferred"),
        ("Benefits & perks", "benefits"),
        ("About us", None),
    ],
)
def test_classify_heading(heading, kind):
    assert classify_heading(heading) == kind


def test_split_sections_handles_markers_bullets_and_inline_headings():
    text = "\n".join(
        [
            "Intro text",
            "## Responsibilities",
            "- Build APIs",
            "## Requirements",
            "- Python",
            "Nice to have:",
            "• Docker",
            "Benefits: Free lunch",
            "## Our story",
            "Ignored narrative",
        ]
    )
    sections = split_sections(text)
    assert sections["about"] == ["Intro text", "Ignored narrative"]
    assert sections["responsibilities"] == ["Build APIs"]
    assert sections["requirements"] == ["Python"]
    assert sections["preferred"] == ["Docker"]
    assert sections["benefits"] == ["Free lunch"]
