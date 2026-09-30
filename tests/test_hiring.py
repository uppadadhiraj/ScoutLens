import pytest

from agents.hiring_agent import HiringAgent
from agents.planner import InvestigationPlanner
from models.opportunity import OpportunityProfile
from models.search import ApplyLink, JobListing
from services.analysis.hiring import (
    analyze_hiring,
    city_of,
    is_same_listing,
    role_similarity,
    seniority_of,
)
from services.evidence.store import EvidenceStore
from tests.fakes import ScriptedClient
from utils.errors import SerpApiError
from utils.language_guard import is_neutral

CO = "Northwind Labs"


def job(n, title="Software Engineer Intern", desc="", loc="Bengaluru, Karnataka, India", company=CO, **kw):
    return JobListing(
        search_id="s1", job_id=f"j{n}", title=title, company_name=company, location=loc, description=desc,
        share_link=f"https://www.google.com/search?q=job{n}", via="LinkedIn", **kw,
    )


def opp(**kw) -> OpportunityProfile:
    base = dict(
        company_name=CO, role_title="Software Engineer Intern", location="Bengaluru, Karnataka, India",
        required_skills=["Python", "SQL"], experience_min_years=0.0, experience_max_years=2.0,
        source_url="https://www.northwindlabs.example/careers/swe-intern",
    )
    return OpportunityProfile(**{**base, **kw})


def analyze(o, listings):
    store = EvidenceStore(CO)
    return analyze_hiring(o, listings, store), store


# ----------------------------------------------------------------- helpers

@pytest.mark.parametrize(
    "title, tier",
    [("Software Engineer Intern", "intern"), ("Senior Backend Engineer", "senior"), ("Staff Engineer", "senior"),
     ("Junior Analyst", "junior"), ("Backend Engineer", "mid"), (None, "mid"), ("Engineering Manager", "senior")],
)
def test_seniority(title, tier):
    assert seniority_of(title) == tier


def test_role_similarity():
    assert role_similarity("Software Engineer Intern", "Software Engineer") == 1.0
    assert role_similarity("Backend Engineer", "Software Engineer") == pytest.approx(1 / 3)
    assert role_similarity("Backend Engineer", "Account Executive") == 0.0
    assert role_similarity(None, "x") == 0.0


def test_city_of():
    assert city_of("Bengaluru, Karnataka, India") == "bengaluru" and city_of(None) == ""


def test_same_listing_by_url_even_when_titles_differ():
    j = job(1, title="Totally Different Title")
    j.apply_links = [ApplyLink(title="Apply", url="https://northwindlabs.example/careers/swe-intern/")]
    assert is_same_listing(j, opp())


def test_same_title_at_another_employer_is_not_the_same_listing():
    # regression: the market sample is full of identical titles from other companies
    assert not is_same_listing(job(1, company="Globex Corp"), opp())
    assert is_same_listing(job(1, company="Northwind Labs Pvt Ltd"), opp())


def test_same_listing_by_title_and_place():
    assert is_same_listing(job(1), opp())
    assert is_same_listing(job(1, loc="Bengaluru"), opp())
    assert not is_same_listing(job(1, loc="Pune, India"), opp())
    assert not is_same_listing(job(1, title="Data Analyst Intern"), opp())


# ----------------------------------------------------------- sample and counts

def test_sample_excludes_other_employers_and_the_original_itself():
    listings = [job(1), job(2, title="Backend Engineer", desc="Python"), job(3, title="Data Engineer", desc="SQL"),
                job(4, title="Recruiter", company="Globex Corp"), job(5, title="Analyst", company="Northwind Labs Pvt Ltd")]
    sig, store = analyze(opp(), listings)
    assert sig.sample_size == 3  # backend, data, analyst (Pvt Ltd variant matches); original and Globex excluded
    assert sig.excluded_other_company == 1
    assert sig.original_found and sig.original_via == "LinkedIn"
    assert any("excluded" in n for n in sig.notes)
    assert all(store.get(i) for i in sig.listing_evidence_ids)


def test_duplicates_across_searches_are_counted_once():
    sig, _ = analyze(opp(), [job(2, title="Backend Engineer"), job(2, title="Backend Engineer")])
    assert sig.sample_size == 1


def test_no_related_listings_is_reported_as_insufficient_evidence():
    sig, _ = analyze(opp(), [job(9, company="Globex Corp")])
    assert sig.sample_size == 0 and not sig.has_data and sig.excluded_other_company == 1
    assert any(n.startswith("Insufficient evidence") for n in sig.notes)
    assert sig.skill_frequencies == [] and sig.comparisons[0].kind == "listing"


def test_empty_results():
    sig, _ = analyze(opp(), [])
    assert sig.sample_size == 0 and sig.comparisons == []


# ------------------------------------------------------------ skill frequencies

def _stack_listings():
    return [
        job(i, title=f"Engineer {i}", desc=d)
        for i, d in enumerate(
            ["Python, SQL and Docker", "Python and Docker on AWS", "Docker and Kubernetes", "Python with AWS",
             "SQL and Docker", "Golang and Kubernetes"], start=2)
    ]


def test_percentages_are_computed_from_the_real_sample_only():
    sig, _ = analyze(opp(), _stack_listings())
    freq = {f.skill: f for f in sig.skill_frequencies}
    assert sig.sample_size == 6
    assert (freq["Docker"].count, freq["Docker"].total, freq["Docker"].percent) == (4, 6, 66.7)
    assert freq["Python"].count == 3 and freq["Python"].percent == 50.0
    assert freq["Kubernetes"].count == 2 and freq["Go"].count == 1
    assert freq["Python"].in_original and not freq["Docker"].in_original
    assert [f.count for f in sig.skill_frequencies] == sorted((f.count for f in sig.skill_frequencies), reverse=True)


def test_a_skill_absent_from_every_listing_is_never_invented():
    sig, _ = analyze(opp(), _stack_listings())
    assert "Terraform" not in {f.skill for f in sig.skill_frequencies}
    assert all(0 < f.count <= f.total == 6 for f in sig.skill_frequencies)


def test_skill_gap_comparison_states_counts_and_cites_the_listings():
    sig, store = analyze(opp(), _stack_listings())
    gaps = [c for c in sig.comparisons if c.kind == "skills" and "Docker" in c.fact]
    assert len(gaps) == 1
    assert "Docker appears in 4 of 6 sampled current listings" in gaps[0].fact and "66.7%" in gaps[0].fact
    assert "not mentioned in this listing" in gaps[0].fact
    assert len(gaps[0].evidence_ids) == 4 and all(store.get(i) for i in gaps[0].evidence_ids)
    # Python is already in the original listing, so it is not called out as a gap
    assert not any("Python appears" in c.fact for c in sig.comparisons)


def test_skill_gap_needs_a_minimum_count():
    sig, _ = analyze(opp(), _stack_listings())
    assert not any("Kubernetes" in c.fact for c in sig.comparisons)  # only 2 of 6 (< 3)


def test_required_skill_absent_from_all_others_needs_a_reasonable_sample():
    o = opp(required_skills=["Python", "Terraform"])
    big, _ = analyze(o, _stack_listings())
    assert any("Terraform is required in this listing but appears in none of the 6" in c.fact for c in big.comparisons)
    small, _ = analyze(o, _stack_listings()[:3])
    assert not any("Terraform" in c.fact for c in small.comparisons)


# --------------------------------------------------------------------- experience

def _exp_listings(*texts, title="Software Engineer Intern"):
    return [job(i, title=f"{title} {i}", desc=f"Requires {t} of experience.") for i, t in enumerate(texts, start=10)]


def test_experience_variation_uses_the_desired_neutral_wording():
    sig, _ = analyze(opp(), _exp_listings("1-3 years", "1-3 years", "2-4 years"))
    (comp,) = [c for c in sig.comparisons if c.kind == "experience"]
    assert "This listing states 0–2 years" in comp.fact
    assert "Among 3 comparable listings" in comp.fact and "1–3 years (2 listings)" in comp.fact and "2–4 years (1 listing)" in comp.fact
    assert "vary across current listings" in comp.interpretation
    assert "does not establish that the original listing is inaccurate" in comp.interpretation
    assert len(comp.evidence_ids) == 3


def test_consistent_experience_is_reported_as_consistent():
    sig, _ = analyze(opp(), _exp_listings("0-2 years", "0-2 years"))
    (comp,) = [c for c in sig.comparisons if c.kind == "experience"]
    assert "same as this one" in comp.fact and "consistent" in comp.interpretation


def test_experience_is_only_compared_within_the_same_seniority_tier():
    listings = _exp_listings("1-3 years", "1-3 years") + _exp_listings("8+ years", "10+ years", title="Senior Staff Engineer")
    sig, _ = analyze(opp(), listings)
    assert len(sig.experience) == 2
    (comp,) = [c for c in sig.comparisons if c.kind == "experience"]
    assert "8" not in comp.fact and "10" not in comp.fact
    assert any("'intern'-level" in n for n in sig.notes)


def test_too_few_comparable_listings_means_no_experience_claim():
    sig, _ = analyze(opp(), _exp_listings("1-3 years"))
    assert not any(c.kind == "experience" for c in sig.comparisons)
    assert any("Insufficient evidence: fewer than 2 comparable" in n for n in sig.notes)


def test_no_experience_comparison_when_the_original_states_none():
    sig, _ = analyze(opp(experience_min_years=None, experience_max_years=None), _exp_listings("1-3 years", "2-4 years"))
    assert not any(c.kind == "experience" for c in sig.comparisons)


# ----------------------------------------------------------- location / mode / salary

def test_location_difference_needs_enough_listings_and_names_where_they_are():
    others = [job(i, title=f"Engineer {i}", loc="Pune, Maharashtra, India") for i in range(20, 24)]
    sig, _ = analyze(opp(), others)
    (comp,) = [c for c in sig.comparisons if c.kind == "location"]
    assert "none of the 4 other sampled listings are there" in comp.fact and "4 are in Pune" in comp.fact
    sig2, _ = analyze(opp(), others[:2])
    assert not any(c.kind == "location" for c in sig2.comparisons)


def test_no_location_comparison_when_the_original_city_is_present_or_remote():
    others = [job(i, title=f"Engineer {i}", loc="Bengaluru, India") for i in range(20, 24)]
    assert not any(c.kind == "location" for c in analyze(opp(), others)[0].comparisons)
    others = [job(i, title=f"Engineer {i}", loc="Pune, India") for i in range(20, 24)]
    assert not any(c.kind == "location" for c in analyze(opp(remote_or_onsite="remote"), others)[0].comparisons)


def test_work_mode_comparison():
    others = [job(i, title=f"Engineer {i}", work_from_home=True) for i in range(20, 24)]
    sig, _ = analyze(opp(remote_or_onsite="onsite"), others)
    (comp,) = [c for c in sig.comparisons if c.kind == "work_mode"]
    assert "This listing is onsite; 4 of 4 other sampled listings state remote work." == comp.fact
    assert sig.work_modes == {"remote": 4}


def test_salary_comparison_only_when_the_original_lacks_one():
    others = [job(i, title=f"Engineer {i}", salary="₹12–18 LPA") for i in range(20, 23)] + [job(30, title="Analyst")]
    sig, _ = analyze(opp(), others)
    (comp,) = [c for c in sig.comparisons if c.kind == "salary"]
    assert "3 of 4 other sampled listings do" in comp.fact and "₹12–18 LPA" in comp.fact
    assert not any(c.kind == "salary" for c in analyze(opp(salary="₹20,000 per month"), others)[0].comparisons)


def test_locations_and_titles_are_summarised():
    listings = [job(2, title="Backend Engineer", loc="Pune, India"), job(3, title="Data Analyst", loc="Pune, India"), job(4, title="QA", loc="Delhi")]
    sig, _ = analyze(opp(location=None), listings)
    assert sig.locations[0].location == "Pune, India" and sig.locations[0].count == 2
    assert sig.titles == ["Backend Engineer", "Data Analyst", "QA"]


# --------------------------------------------------------- listing corroboration

def test_original_listing_found_and_not_found_are_both_stated_neutrally():
    found, _ = analyze(opp(), [job(1), job(2, title="Backend Engineer")])
    (c1,) = [c for c in found.comparisons if c.kind == "listing"]
    assert c1.fact == "This listing also appears in Google Jobs (via LinkedIn)." and c1.evidence_ids
    missing, _ = analyze(opp(), [job(2, title="Backend Engineer"), job(3, title="Data Engineer")])
    (c2,) = [c for c in missing.comparisons if c.kind == "listing"]
    assert "was not among the 2 results" in c2.fact and "does not show the listing is inactive or inaccurate" in c2.interpretation


# ----------------------------------------------------------- language and citations

def test_every_generated_sentence_is_neutral_and_every_comparison_is_cited_or_search_level():
    listings = _stack_listings() + _exp_listings("1-3 years", "2-4 years") + [
        job(40, title="Engineer 40", loc="Pune, India", work_from_home=True, salary="₹10 LPA"),
        job(41, title="Engineer 41", loc="Pune, India", work_from_home=True, salary="₹11 LPA"),
        job(42, title="Engineer 42", loc="Pune, India", work_from_home=True, salary="₹12 LPA"),
    ]
    sig, store = analyze(opp(remote_or_onsite="onsite", required_skills=["Python", "Terraform"]), listings)
    # (no "location" here: most other listings are in the original's own city, so there is nothing to report)
    assert {c.kind for c in sig.comparisons} >= {"volume", "skills", "experience", "work_mode", "salary"}
    for c in sig.comparisons:
        assert is_neutral(f"{c.fact} {c.interpretation} {c.action or ''}"), c
        assert c.evidence_ids, f"uncited comparison: {c.fact}"
        assert all(store.get(i) for i in c.evidence_ids)


# ------------------------------------------------------------------- the agent

@pytest.fixture
def planned(settings):
    return InvestigationPlanner(settings).initial_plan(opp())


def test_agent_collects_searches_and_reports_the_executed_search_ids(settings, planned, jobs_payload):
    routes = {("google_jobs", "northwind labs jobs"): jobs_payload, ("google_jobs", "software engineer intern northwind labs"): {}}
    client = ScriptedClient(settings, routes)
    agent = HiringAgent(client, EvidenceStore(CO))
    steps = [planned.step("jobs_company"), planned.step("jobs_role")]
    sig = agent.investigate(opp(location=None), steps)

    assert sig.sample_size == 1  # job-1 (Backend Engineer); job-2 is the original (same title, no place to contradict)
    assert sig.original_found
    # the company search follows its pagination token (pages=2); the role search is scoped to Bengaluru, finds
    # nothing there, and so runs a second time without a location (regional fallback)
    assert sig.search_ids == ["s1", "s2", "s3", "s4"]
    assert [r.query for r in client.records] == [
        "Northwind Labs jobs", "Northwind Labs jobs", "Software Engineer Intern Northwind Labs", "Software Engineer Intern Northwind Labs"]
    assert any("has no listings for 'Bengaluru, Karnataka, India'" in n for n in sig.notes)


def test_agent_survives_a_non_fatal_search_failure(settings, planned, jobs_payload):
    client = ScriptedClient(
        settings,
        {("google_jobs", "northwind labs jobs"): jobs_payload},
        errors={("google_jobs", "software engineer intern northwind labs"): SerpApiError(kind="timeout")},
    )
    sig = HiringAgent(client, EvidenceStore(CO)).investigate(opp(location=None), [planned.step("jobs_company"), planned.step("jobs_role")])
    assert sig.sample_size == 1
    assert any("failed" in n for n in sig.notes)


def test_agent_propagates_fatal_errors(settings, planned):
    client = ScriptedClient(settings, errors={("google_jobs", "northwind labs jobs"): SerpApiError(kind="invalid_key")})
    with pytest.raises(SerpApiError) as info:
        HiringAgent(client, EvidenceStore(CO)).investigate(opp(), [planned.step("jobs_company")])
    assert info.value.is_fatal


# ---- regional fallback: Google Jobs coverage is regional and an empty response is not an error ---------------

def test_a_different_region_sample_skips_comparisons_that_would_be_meaningless():
    others = [job(i, title=f"Engineer {i}", loc="Mumbai, India", work_from_home=True, salary="₹12–18 LPA") for i in range(20, 25)]
    same, _ = analyze(opp(remote_or_onsite="onsite"), others)
    assert {"location", "work_mode", "salary"} <= {c.kind for c in same.comparisons}

    store = EvidenceStore(CO)
    other = analyze_hiring(opp(remote_or_onsite="onsite"), others, store, same_region=False, search_region="India")
    kinds = {c.kind for c in other.comparisons}
    assert not ({"location", "work_mode", "salary"} & kinds)
    assert other.search_region == "India" and any("different region (India)" in n and "would not be comparable" in n for n in other.notes)
    assert "skills" in kinds or "volume" in kinds  # region-independent comparisons remain


def _regional(payload_for_default):
    """Route that returns nothing for a location-scoped request and `payload_for_default` otherwise."""
    return lambda params: {} if params.get("location") == "Stockholm" else payload_for_default


def test_empty_regional_search_falls_back_to_the_default_region_and_says_so(settings, jobs_payload):
    from models.search import SearchQuery
    from services.serpapi.jobs import fetch_jobs_with_fallback

    client = ScriptedClient(settings, {("google_jobs", "acme jobs"): _regional(jobs_payload)})
    result = fetch_jobs_with_fallback(client, SearchQuery(engine="google_jobs", q="acme jobs", location="Stockholm"), "India")
    assert result.fell_back and result.location_used == "India" and result.requested_location == "Stockholm"
    assert len(result.listings) == 2
    assert [p.get("location") for p in client.params_seen] == ["Stockholm", "India"]  # second search used the configured default


def test_no_fallback_when_the_regional_search_has_results_or_is_already_the_default(settings, jobs_payload):
    from models.search import SearchQuery
    from services.serpapi.jobs import fetch_jobs_with_fallback

    ok = ScriptedClient(settings, {("google_jobs", "acme jobs"): jobs_payload})
    r1 = fetch_jobs_with_fallback(ok, SearchQuery(engine="google_jobs", q="acme jobs", location="Pune"), "India")
    assert not r1.fell_back and r1.location_used == "Pune" and len(ok.requested) == 1  # one page, no fallback

    empty = ScriptedClient(settings, {})
    r2 = fetch_jobs_with_fallback(empty, SearchQuery(engine="google_jobs", q="x", location="india"), "India")
    assert not r2.fell_back and len(empty.requested) == 1  # already the default region: retrying would be pointless

    r3 = fetch_jobs_with_fallback(ScriptedClient(settings, {}), SearchQuery(engine="google_jobs", q="x"), "India")
    assert not r3.fell_back and r3.location_used == "India"


def test_hiring_agent_reports_the_fallback_and_drops_regional_comparisons(settings, jobs_payload):
    client = ScriptedClient(settings, {("google_jobs", "northwind labs jobs"): _regional(jobs_payload)})
    plan = InvestigationPlanner(settings).initial_plan(opp(location="Stockholm; London"))
    sig = HiringAgent(client, EvidenceStore(CO), "India").investigate(opp(location="Stockholm; London"), [plan.step("jobs_company")])
    assert sig.sample_size == 2 and sig.search_region == "India"
    assert any("has no listings for 'Stockholm'" in n and "searched in 'India' instead" in n for n in sig.notes)
    assert not {"location", "work_mode", "salary"} & {c.kind for c in sig.comparisons}


# ---- lessons from real Indian data --------------------------------------------------------------------

def test_bangalore_and_bengaluru_are_the_same_city():
    from services.analysis.hiring import city_of

    assert city_of("Bangalore, Karnataka") == city_of("Bengaluru, Karnataka, India") == "bengaluru"
    assert city_of("Bombay") == "mumbai" and city_of("Gurgaon, Haryana") == "gurugram" and city_of("New Delhi") == "delhi"


def test_no_false_location_finding_when_the_listing_uses_an_alternative_city_name():
    others = [job(i, title=f"Engineer {i}", loc="Bengaluru, Karnataka, India") for i in range(20, 24)]
    sig, _ = analyze(opp(location="Bangalore, Karnataka"), others)
    assert not any(c.kind == "location" for c in sig.comparisons)


def test_experience_for_senior_roles_is_compared_only_with_similar_titles():
    # a Security Architect (10+ years) is not comparable with every other senior role in the company
    security = opp(role_title="Security Architect", experience_min_years=10.0, experience_max_years=None)
    listings = [job(i, title=t, desc=f"Requires {y} of experience.") for i, (t, y) in enumerate(
        [("Senior Sales Manager", "6+ years"), ("Senior Data Engineer", "4-7 years"), ("Senior Marketing Lead", "8+ years"),
         ("Senior Security Engineer", "5+ years"), ("Senior Security Architect", "12+ years")], start=50)]
    sig, _ = analyze(security, listings)
    assert [o.title for o in sig.experience] == ["Senior Security Engineer", "Senior Security Architect"]
    assert any("similar title" in n for n in sig.notes)
    (comp,) = [c for c in sig.comparisons if c.kind == "experience"]
    assert "6+" not in comp.fact and "4–7" not in comp.fact


def test_entry_level_experience_is_still_compared_across_departments():
    sig, _ = analyze(opp(), _exp_listings("1-3 years", "1-3 years"))
    assert len(sig.experience) == 2 and not any("similar title" in n for n in sig.notes)
