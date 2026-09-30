import pytest

from models.opportunity import OpportunityProfile
from models.resume import ResumeProfile, ResumeSkill, SkillEvidence
from services.resume.matching import match_skills
from services.resume.parser import parse_resume_text
from utils.language_guard import is_neutral

RESUME = """Asha Verma

Summary
Student who enjoys backend work.

Education
B.Tech in Computer Science, Example Institute, 2026
Coursework: Data Structures, Machine Learning basics

Skills
Languages: Python, SQL
Frameworks: FastAPI
Databases: Postgres
Tools: Git, Docker, Azure

Projects
Tracker | Flask
• Built an API with scikit-learn based recommendations

Experience
Intern - Acme Corp
• Deployed services with Kubernetes on GCP and wrote Terraform modules
"""


@pytest.fixture(scope="module")
def resume() -> ResumeProfile:
    return parse_resume_text(RESUME)


def opp(required=(), preferred=(), **kw) -> OpportunityProfile:
    return OpportunityProfile(company_name="Acme", role_title="Engineer", required_skills=list(required), preferred_skills=list(preferred), **kw)


def status_of(summary, skill):
    return next(m for m in summary.matches if m.skill == skill)


def test_listed_and_used_skills_are_matched_with_a_basis(resume):
    fit = match_skills(opp(["Python", "Kubernetes"]), resume)
    python, k8s = status_of(fit, "Python"), status_of(fit, "Kubernetes")
    assert (python.status, python.basis) == ("matched", "Listed in the Skills section")
    assert k8s.status == "matched" and k8s.basis == "Used in Experience on your resume"
    assert k8s.resume_evidence and "Kubernetes" in k8s.resume_evidence[0].snippet


def test_implied_skills_are_matched_and_say_why(resume):
    fit = match_skills(opp(["SQL", "Python"]), parse_resume_text("Jane\n\nSkills\nPostgres, FastAPI, Git, Docker, Linux\n"))
    sql = status_of(fit, "SQL")
    assert sql.status == "matched" and sql.related_skill == "PostgreSQL" and sql.basis == "PostgreSQL on your resume involves SQL"
    assert status_of(fit, "Python").related_skill == "FastAPI"


def test_mentioned_only_in_coursework_is_partial_not_matched(resume):
    fit = match_skills(opp(["Data Structures"]), resume)
    m = status_of(fit, "Data Structures")
    assert m.status == "partial" and "Mentioned only in Education" in m.basis


def test_related_but_different_tool_is_partial(resume):
    fit = match_skills(opp(["AWS", "Node.js"]), resume)
    aws = status_of(fit, "AWS")
    assert aws.status == "partial" and aws.related_skill in {"Azure", "GCP"} and "different tool" in aws.basis
    assert status_of(fit, "Node.js").status == "not_found"


def test_machine_learning_is_partial_via_a_library(resume):
    # the resume mentions "Machine Learning basics" in coursework AND uses scikit-learn in a project
    fit = match_skills(opp(["Machine Learning"]), resume)
    assert status_of(fit, "Machine Learning").status == "partial"


def test_absence_is_phrased_as_not_found_in_resume_never_as_lack_of_knowledge(resume):
    fit = match_skills(opp(["Rust"]), resume)
    m = status_of(fit, "Rust")
    assert m.status == "not_found" and m.basis == "Not found in resume"
    for match in fit.matches:
        assert "don't know" not in match.basis.lower() and "lack" not in match.basis.lower()
        assert is_neutral(match.basis)


def test_no_matching_by_string_similarity():
    resume = parse_resume_text(RESUME.replace("Languages: Python, SQL", "Languages: JavaScript, MySQL"))
    assert resume.quality == "good"
    fit = match_skills(opp(["Java", "SQL"]), resume)
    assert status_of(fit, "Java").status == "not_found"  # JavaScript is not Java
    assert status_of(fit, "SQL").status == "matched"  # MySQL implies SQL, but only because it is declared in IMPLIES


def test_sparse_resume_yields_unknown_instead_of_not_found():
    resume = parse_resume_text("Jane Roe\n\nSkills\nPython\n")
    fit = match_skills(opp(["Python", "Docker"]), resume)
    assert status_of(fit, "Python").status == "matched"
    docker = status_of(fit, "Docker")
    assert docker.status == "unknown" and "too limited" in docker.basis
    assert fit.resume_quality == "limited" and any("unknown rather than not found" in n for n in fit.notes)


def test_coverage_arithmetic_is_transparent(resume):
    fit = match_skills(opp(["Python", "SQL", "Docker", "Rust"], preferred=["Terraform", "Go"]), resume)
    assert (fit.total_requirements, fit.matched, fit.partial, fit.not_found, fit.unknown) == (6, 4, 0, 2, 0)
    assert fit.coverage_percent == 66.7  # 4 of 6
    assert (fit.required_total, fit.required_matched, fit.required_coverage_percent) == (4, 3, 75.0)
    assert [m.requirement for m in fit.matches] == ["required"] * 4 + ["preferred"] * 2


def test_partial_matches_do_not_inflate_coverage(resume):
    fit = match_skills(opp(["AWS", "Python"]), resume)
    assert fit.partial == 1 and fit.matched == 1 and fit.coverage_percent == 50.0


def test_a_skill_in_both_lists_is_counted_once():
    fit = match_skills(opp(["Python"], preferred=["Python", "Docker"]), parse_resume_text("Jane\n\nSkills\nPython\n"))
    assert [m.skill for m in fit.matches] == ["Python", "Docker"]


def test_empty_requirements():
    fit = match_skills(opp(), parse_resume_text("Jane\n\nSkills\nPython, SQL, Git, Docker, Linux\n"))
    assert fit.total_requirements == 0 and fit.coverage_percent is None and fit.required_coverage_percent is None
    assert any("nothing to compare" in n for n in fit.notes)


def test_self_reported_skills_match():
    resume = ResumeProfile(quality="good", skills=[ResumeSkill(name="Docker", category="devops", level="self_reported", evidence=[SkillEvidence(section="Preferences", snippet="Entered by you")])])
    m = status_of(match_skills(opp(["Docker"]), resume), "Docker")
    assert m.status == "matched" and m.basis == "Entered by you in the preferences"


def test_context_lines_are_side_by_side_facts_without_judgement(resume):
    fit = match_skills(
        opp(["Python"], experience_required="0-2 years", experience_min_years=0, experience_max_years=2, education_required="Pursuing a B.Tech"),
        resume,
    )
    assert fit.context[0] == "The listing states its experience requirement as '0-2 years' (0–2 years). Your resume lists 1 experience entry and 1 project."
    assert "Pursuing a B.Tech" in fit.context[1] and "B.Tech in Computer Science" in fit.context[1]
    assert all(is_neutral(line) for line in fit.context)
