import pytest

from services.skills import extract_skills, normalize_skill, skill_category


@pytest.mark.parametrize(
    "text",
    ["PostgreSQL", "Postgres", "postgresql database", "  POSTGRES. "],
)
def test_postgres_variants_normalize_to_one_skill(text):
    assert normalize_skill(text) == "PostgreSQL"


@pytest.mark.parametrize("text", ["Amazon Web Services", "aws", "AWS"])
def test_aws_variants(text):
    assert normalize_skill(text) == "AWS"


def test_normalize_does_not_guess_by_similarity():
    assert normalize_skill("Postgre") is None
    assert normalize_skill("PostgreSQLish") is None
    assert normalize_skill("teamwork") is None


def test_java_is_not_javascript_and_vice_versa():
    assert extract_skills("We use JavaScript and React") == ["JavaScript", "React"]
    assert extract_skills("Strong Java skills") == ["Java"]
    assert extract_skills("Java, JavaScript") == ["Java", "JavaScript"]


def test_sql_is_not_matched_inside_nosql_or_mysql():
    assert extract_skills("Experience with NoSQL stores") == ["NoSQL"]
    assert extract_skills("MySQL administration") == ["MySQL"]
    assert "SQL" in extract_skills("MySQL and SQL")


def test_special_character_skills():
    found = extract_skills("Proficient in C++, C#, Node.js and .NET; ASP.NET a plus")
    assert {"C++", "C#", "Node.js", ".NET", "ASP.NET"} <= set(found)
    assert "C" not in found


def test_ambiguous_words_need_context():
    # Ordinary English must not be mistaken for languages/tools, even when capitalised at a sentence start.
    for english in (
        "Go to market strategy; we excel at teamwork; react quickly to change",
        "Swift execution and rust on the pipes",
        "React quickly to changing customer needs",
        "Excel in a fast-paced environment",
        "Spark curiosity across the team",
    ):
        found = extract_skills(english)
        assert not {"Go", "Swift", "Rust", "React", "Excel", "Apache Spark"} & set(found), english
    # ...but the real thing is recognised.
    assert "Go" in extract_skills("Languages: Python, Go, Rust")
    assert "Rust" in extract_skills("Languages: Python, Go, Rust")
    assert "Go" in extract_skills("We write Golang services")
    assert "R" in extract_skills("Statistics with R programming")
    assert "Excel" in extract_skills("Advanced Excel and Power BI")
    assert {"iOS", "Swift"} <= set(extract_skills("iOS apps in Swift"))
    assert "Swift" in extract_skills("Kotlin, Swift, Dart")
    assert {"React", "TypeScript"} <= set(extract_skills("React and TypeScript"))
    assert "Apache Spark" in extract_skills("Hadoop, Spark, Kafka")
    assert "Snowflake" in extract_skills("Snowflake and BigQuery warehouses")


def test_dot_js_suffix_does_not_imply_javascript():
    assert extract_skills("Node.js and React.js") == ["Node.js", "React"]


def test_case_rules_match_the_spelling_they_protect():
    assert "Rust" not in extract_skills("rust on the pipes")
    assert "Rust" in extract_skills("Rust")
    assert "Docker" in extract_skills("DOCKER") and "Docker" in extract_skills("docker")
    assert normalize_skill("golang") == "Go" and normalize_skill("GOLANG") == "Go"


def test_extraction_is_ordered_by_first_appearance_and_deduplicated():
    assert extract_skills("Docker then Python, then docker again and Python") == ["Docker", "Python"]


def test_empty_text():
    assert extract_skills("") == []


def test_categories():
    assert skill_category("Docker") == "devops"
    assert skill_category("Nope") is None


def test_security_roles_are_recognised_from_their_own_vocabulary():
    assert "Cybersecurity" in extract_skills("AI-powered security engineering for the platform")
    assert "Cybersecurity" in extract_skills("You will run threat modeling and vulnerability management")
    assert "Cybersecurity" not in extract_skills("We value security of our customers' data")  # a bare word is not a skill
