import pytest

from services.resume import pdf as pdf_module
from services.resume.parser import add_self_reported_skills, parse_resume_text
from tests.pdf_factory import blank_pdf, encrypted_pdf, make_pdf
from utils.errors import ResumeParseError

RESUME = """ASHA VERMA
asha.verma@example.com | +91 98765 43210 | github.com/ashaverma | linkedin.com/in/ashaverma

Summary
Final-year computer science student interested in backend development. I excel at teamwork.

Education
B.Tech in Computer Science, Example Institute of Technology, 2022 - 2026
CGPA: 8.4/10

Technical Skills
Languages: Python, Java, SQL, JavaScript
Frameworks: FastAPI, Flask, React
Databases: Postgres, MongoDB
Tools: Git, Docker, Linux

Projects
Expense Tracker | Python, Flask
• Built a REST API with Flask and PostgreSQL database
• Added unit tests with pytest
Chat App | React, Node.js
• Real-time chat using WebSockets and Redis

Experience
Backend Intern - Acme Corp (Jun 2025 - Aug 2025)
• Developed internal tools in Python and deployed on AWS
• Wrote SQL reports

Certifications
• AWS Cloud Practitioner
• Machine Learning by Andrew Ng (Coursera)

Achievements
• Winner, Smart India Hackathon 2024 internal round
"""


@pytest.fixture(scope="module")
def profile():
    return parse_resume_text(RESUME)


# ---------------------------------------------------------------------- parsing

def test_name_and_sections(profile):
    assert profile.name == "Asha Verma"  # all-caps normalised; contact line is not a name
    assert profile.sections_found == ["summary", "education", "skills", "projects", "experience", "certifications", "achievements"]
    assert profile.quality == "good" and profile.notes == []


def test_structured_sections(profile):
    assert profile.education[0].startswith("B.Tech in Computer Science")
    assert [p.title for p in profile.projects] == ["Expense Tracker | Python, Flask", "Chat App | React, Node.js"]
    assert profile.projects[0].details == ["Built a REST API with Flask and PostgreSQL database", "Added unit tests with pytest"]
    assert [e.title for e in profile.experience] == ["Backend Intern - Acme Corp (Jun 2025 - Aug 2025)"]
    assert profile.certifications == ["AWS Cloud Practitioner", "Machine Learning by Andrew Ng (Coursera)"]
    assert profile.achievements == ["Winner, Smart India Hackathon 2024 internal round"]


def test_skills_are_normalised_and_categorised(profile):
    assert profile.programming_languages == ["Java", "JavaScript", "Python", "SQL"]
    assert {"FastAPI", "Flask", "React", "Node.js"} <= set(profile.frameworks)
    assert {"PostgreSQL", "MongoDB", "Docker", "Git", "Linux", "AWS"} <= set(profile.tools)
    assert "Postgres" not in profile.skill_names  # canonical name only


def test_skill_levels_reflect_where_it_was_found(profile):
    assert profile.skill("Python").level == "listed"
    assert profile.skill("AWS").level == "used"  # experience + certification, not in the skills section
    assert profile.skill("Redis").level == "used"
    # a course certificate shows learning, not application, so it is only a mention
    assert profile.skill("Machine Learning").level == "mentioned"


def test_skill_evidence_shows_where(profile):
    aws = profile.skill("AWS")
    assert {e.section for e in aws.evidence} == {"Experience", "Certifications"}
    assert aws.level == "used"  # the strongest evidence wins
    assert any("deployed on AWS" in e.snippet for e in aws.evidence)
    pg = profile.skill("PostgreSQL")
    assert pg.level == "listed" and {"Skills", "Projects"} <= {e.section for e in pg.evidence}


def test_urls_and_emails_are_not_mistaken_for_skills(profile):
    assert "GitHub" not in profile.skill_names  # github.com/ashaverma is a contact link


def test_ordinary_english_is_not_a_skill(profile):
    assert "Excel" not in profile.skill_names  # "I excel at teamwork"


def test_a_skill_absent_from_the_resume_is_absent(profile):
    assert profile.skill("Kubernetes") is None and profile.skill("Terraform") is None


def test_skills_section_tokens_handle_labels_parentheses_and_short_names():
    p = parse_resume_text("Jane Roe\n\nSkills\nLanguages: Python (Pandas, NumPy), R, Go\nOther: Tableau | Power BI\n")
    assert {"Python", "Pandas", "NumPy", "R", "Go", "Tableau", "Power BI"} <= p.skill_names
    assert all(p.skill(n).level == "listed" for n in ("Python", "R", "Go"))


def test_resume_without_headings_still_yields_skills_and_a_note():
    p = parse_resume_text("Jane Roe\nBuilt services in Python and Docker.\nDeployed on AWS with Terraform.\n")
    assert {"Python", "Docker", "AWS", "Terraform"} <= p.skill_names
    assert any("No 'Skills' section" in n for n in p.notes)
    assert p.sections_found == []


def test_sparse_resume_is_flagged_as_limited():
    p = parse_resume_text("Jane Roe\n\nSkills\nPython\n")
    assert p.quality == "limited"
    assert any("may not be meaningful" in n for n in p.notes)


@pytest.mark.parametrize(
    "first_line, expected",
    [("Software Engineer", None), ("Curriculum Vitae", None), ("R. K. Narayan", "R. K. Narayan"),
     ("PRIYA SHARMA", "Priya Sharma"), ("Mary-Jane O'Neil", "Mary-Jane O'Neil")],
)
def test_name_detection(first_line, expected):
    assert parse_resume_text(f"{first_line}\n\nSkills\nPython\n").name == expected


def test_self_reported_skills_are_merged_and_unknowns_reported():
    p = parse_resume_text("Jane Roe\n\nSkills\nPython\n")
    add_self_reported_skills(p, ["postgres", "Python", "Underwater Basket Weaving"])
    assert p.skill("PostgreSQL").level == "self_reported"
    assert p.skill("Python").level == "listed"  # already listed: level not downgraded
    assert any(e.section == "Preferences" for e in p.skill("Python").evidence)
    assert any("not recognised" in n for n in p.notes)


# -------------------------------------------------------------------------- PDF

def _resume_pdf() -> bytes:
    return make_pdf(RESUME.splitlines())


def test_pdf_round_trip_produces_the_same_profile_shape():
    text = pdf_module.extract_text(_resume_pdf())
    p = parse_resume_text(text)
    assert p.name == "Asha Verma"
    assert {"Python", "FastAPI", "PostgreSQL", "AWS", "Docker"} <= p.skill_names
    assert "skills" in p.sections_found and len(p.projects) == 2


def test_multi_page_pdf():
    lines = RESUME.splitlines() * 2
    assert "Certifications" in pdf_module.extract_text(make_pdf(lines, per_page=30))


@pytest.mark.parametrize(
    "data, needle",
    [
        (b"", "empty"),
        (b"PK\x03\x04 this is a zip", "not a PDF"),
        (b"<html>%PDF- no</html>".replace(b"%PDF-", b"plain"), "not a PDF"),
    ],
)
def test_invalid_uploads_are_rejected_with_clear_messages(data, needle):
    with pytest.raises(ResumeParseError) as info:
        pdf_module.extract_text(data)
    assert needle in info.value.user_message


def test_oversized_upload_is_rejected_before_parsing():
    with pytest.raises(ResumeParseError, match="larger than"):
        pdf_module.extract_text(_resume_pdf(), max_mb=0.001)


def test_scanned_or_blank_pdf_gets_a_specific_message():
    with pytest.raises(ResumeParseError) as info:
        pdf_module.extract_text(blank_pdf())
    assert "No readable text" in info.value.user_message


def test_encrypted_pdf_asks_for_an_unprotected_copy():
    with pytest.raises(ResumeParseError) as info:
        pdf_module.extract_text(encrypted_pdf(RESUME.splitlines()))
    assert "password" in info.value.user_message


def test_too_many_pages_is_rejected():
    with pytest.raises(ResumeParseError, match="more than 10 pages"):
        pdf_module.extract_text(blank_pdf(pages=11))


def test_corrupt_pdf_never_raises_a_raw_library_error():
    corrupt = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 99 0 R >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"
    with pytest.raises(ResumeParseError) as info:
        pdf_module.extract_text(corrupt)
    assert info.value.user_message  # safe, user-facing text
