"""UI tests: HTML builders (XSS safety) and full flows through Streamlit's AppTest harness."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
from models.evidence import EvidenceItem
from models.report import Finding
from models.resume import SkillMatch
from services.pipeline import NeedsCompanyError
from ui import components as c
from ui.styles import badge, chips, esc, link, safe_url
from utils.errors import ExtractionError, InvalidURLError, SerpApiError

APP = str(Path(__file__).resolve().parent.parent / "app.py")
EVIL = "<script>alert('x')</script><img src=x onerror=alert(1)>"


# ------------------------------------------------------------------ escaping

def test_esc_neutralises_markup_and_quotes():
    assert "<" not in esc(EVIL) and esc('a"b') == "a&quot;b" and esc(None) == ""


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,<script>", "ftp://x", "", None, "//evil.example"])
def test_only_http_urls_become_links(url):
    assert safe_url(url) is None
    assert "<a " not in link("click", url)


def test_links_are_escaped_and_hardened():
    html = link("t", 'https://x.example/?a="><script>alert(1)</script>')
    assert "<script>" not in html and 'rel="noopener noreferrer"' in html and 'target="_blank"' in html


def test_badges_and_chips_escape_untrusted_text():
    assert "<script>" not in badge(EVIL, "ok") + chips([EVIL])


def _evidence(**kw) -> EvidenceItem:
    base = dict(id="E1", claim=EVIL, source_title=EVIL, source_url="https://x.example/a", source_type="web", publisher=EVIL,
                date=EVIL, evidence_text=EVIL, relevance="high", confidence="high", tier=1, tier_label=EVIL, search_id="s1",
                retrieved_at=datetime(2026, 9, 30, tzinfo=timezone.utc))
    return EvidenceItem(**{**base, **kw})


def test_untrusted_content_never_reaches_the_page_as_markup():
    item = _evidence()
    finding = Finding(kind="skills", icon="🔎", title=EVIL, fact=EVIL, interpretation=EVIL, action=EVIL, evidence_ids=["E1"], sources_summary=EVIL)
    match = SkillMatch(skill=EVIL, requirement="required", status="matched", basis=EVIL)
    from models.report import NewsHighlight

    news = NewsHighlight(title=EVIL, url="https://x.example/n", publisher=EVIL, summary=EVIL, topic=EVIL, relevance="high",
                         relevance_reason=EVIL, evidence_id="E1", tier=2)
    rendered = [c.evidence_card(item), c.finding_card(finding, {"E1": item}), c.fit_row(match), c.news_card(news),
                c.cited_text(f"{EVIL} [E1] [E9]", {"E1": item}), c.notice(EVIL), c.demo_banner(EVIL),
                c.steps_html([("k", EVIL, "done", EVIL)]), c.kv_grid([(EVIL, EVIL)])]
    for html in rendered:
        assert "<script" not in html and "<img" not in html, html[:200]


def test_citation_markers_link_to_the_real_source_only_when_the_id_exists():
    item = _evidence(claim="c", source_title="t", publisher=None, date=None, evidence_text="e", tier_label="Official")
    html = c.cited_text("Fact [E1], invented [E7].", {"E1": item})
    assert 'href="https://x.example/a"' in html and "[E7]" in html and html.count("<a ") == 1


def test_fit_row_does_not_repeat_itself_for_not_found():
    row = c.fit_row(SkillMatch(skill="Linux", requirement="preferred", status="not_found", basis="Not found in resume"))
    assert row.count("Not found in resume") == 1
    partial = c.fit_row(SkillMatch(skill="ML", requirement="preferred", status="partial", basis="Mentioned only in Certifications"))
    assert "<b>Partial.</b>" in partial


# --------------------------------------------------------------------- flows

@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "ui.db"))
    monkeypatch.setenv("SERPAPI_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    config.get_settings.cache_clear()
    st.cache_resource.clear()
    yield AppTest.from_file(APP, default_timeout=90).run()
    config.get_settings.cache_clear()
    st.cache_resource.clear()


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def all_text(at) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning]
    parts += [s.value for s in at.success] + [i.value for i in at.info]
    return "\n".join(parts)


def test_home_explains_missing_search_configuration_and_offers_the_demo(app):
    assert not app.exception
    text = all_text(app)
    assert "Live search is not configured" in text and "Investigate the opportunity" in text
    assert [b.label for b in app.button] == ["Investigate", "Try the demo"]


def test_demo_flow_end_to_end(app):
    click(app, "Try the demo")
    assert not app.exception and app.session_state["view"] == "investigation"
    assert "Investigation complete" in all_text(app) and "Demo data" in all_text(app)
    click(app, "Open report →")
    assert not app.exception and app.session_state["view"] == "report"
    assert [t.label for t in app.tabs] == ["Overview", "Company", "Hiring & market", "Personal fit", "Evidence & method"]
    text = all_text(app)
    for expected in ("What you might have missed", "Suggested next actions", "Investigation method", "Demo data"):
        assert expected in text
    assert "23 of 61 sampled listings mention Docker (37.7%)." in text
    assert "AWS appears in 10 of 18 sampled current listings" in text
    assert len(app.checkbox) >= 5  # the neutral action checklist


def test_report_is_saved_to_history_and_can_be_reopened_and_deleted(app):
    click(app, "Try the demo")
    app.session_state["view"] = "history"
    app.run()
    assert "Software Engineer Intern" in all_text(app) and "Demo data" in all_text(app)
    click(app, "Open")
    assert app.session_state["view"] == "report" and not app.exception
    app.session_state["view"] = "history"
    app.run()
    click(app, "Delete")
    assert "No investigations yet" in all_text(app)


def test_history_is_empty_at_first(app):
    app.session_state["view"] = "history"
    app.run()
    assert "No investigations yet" in all_text(app) and not app.exception


def test_report_view_without_a_report_guides_the_user(app):
    app.session_state["view"] = "report"
    app.run()
    assert "No report is open" in all_text(app) and not app.exception


def _submit(app, url="", pasted="", company=""):
    fields = {t.label: t for t in app.text_input}
    if url:
        fields["Paste an opportunity URL"].set_value(url)
    if company:
        fields["Company name"].set_value(company)
    if pasted:
        app.text_area[0].set_value(pasted)
    click(app, "Investigate")


def test_submitting_nothing_shows_a_helpful_message(app):
    _submit(app)
    assert "Paste an opportunity URL, or paste the job description." in all_text(app) and app.session_state["view"] == "home"


def test_pasting_a_description_requires_a_company(app):
    _submit(app, pasted="Software Engineer Intern with Python and SQL, building internal tools for the team.")
    assert "Please enter the company name" in all_text(app) and app.session_state["view"] == "home"


def test_bad_url_returns_home_with_the_error_and_the_paste_fallback_open(app, monkeypatch):
    _submit(app, url="javascript:alert(1)")
    assert not app.exception and app.session_state["view"] == "home"
    assert "valid public http(s) URL" in all_text(app)
    assert app.expander[-1].label.startswith("Can’t open the link") and app.expander[-1].proto.expanded


def test_unreadable_page_offers_manual_paste(app, monkeypatch):
    def fail(*args, **kwargs):
        raise ExtractionError()

    monkeypatch.setattr("ui.investigation.run_pipeline", fail)
    _submit(app, url="https://example.com/job")
    assert "paste the job description manually" in all_text(app) and app.session_state["view"] == "home"


def test_missing_company_flow_asks_for_it_and_then_continues(app, monkeypatch):
    from services.demo import demo_opportunity
    from utils.errors import ScoutLensError

    opp = demo_opportunity(datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)).model_copy(update={"company_name": None})
    submitted = []

    def pipeline(request, *args, **kwargs):
        submitted.append(request)
        if request.opportunity is None:
            raise NeedsCompanyError(opp)
        raise ScoutLensError("stop here")  # the second call is all we need to inspect

    monkeypatch.setattr("ui.investigation.run_pipeline", pipeline)
    _submit(app, url="https://example.com/job")
    assert not app.exception and "company name could not be found" in all_text(app)
    assert [b.label for b in app.button] == ["Continue investigation", "Start over"]

    app.text_input[0].set_value("Lumenpath Labs")
    click(app, "Continue investigation")
    assert len(submitted) == 2
    assert submitted[1].opportunity is opp and submitted[1].company == "Lumenpath Labs"  # nothing is re-fetched or lost
    assert "stop here" in all_text(app) and not app.exception


def test_start_over_clears_the_recovery_panel(app, monkeypatch):
    from services.demo import demo_opportunity

    opp = demo_opportunity(datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)).model_copy(update={"company_name": None})
    monkeypatch.setattr("ui.investigation.run_pipeline", lambda *a, **k: (_ for _ in ()).throw(NeedsCompanyError(opp)))
    _submit(app, url="https://example.com/job")
    click(app, "Start over")
    assert [b.label for b in app.button] == ["Investigate", "Try the demo"] and not app.exception


def test_missing_serpapi_key_shows_setup_help_not_a_traceback(app, monkeypatch):
    def no_key(*args, **kwargs):
        raise SerpApiError(kind="no_key")

    monkeypatch.setattr("ui.investigation.run_pipeline", no_key)
    _submit(app, url="https://example.com/job")
    text = all_text(app)
    assert "SERPAPI_KEY" in text and "try the demo" in text and "Traceback" not in text and not app.exception


def test_an_unexpected_bug_shows_a_generic_message_without_details(app, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail /etc/passwd")

    monkeypatch.setattr("ui.investigation.run_pipeline", boom)
    _submit(app, url="https://example.com/job")
    text = all_text(app)
    assert "Something went wrong while investigating" in text and "secret internal detail" not in text and not app.exception


def test_a_crashing_view_is_caught_by_the_router(app, monkeypatch):
    def broken():
        raise RuntimeError("view exploded: api_key=abc123")

    monkeypatch.setattr("ui.home.render", broken)
    app.run()
    text = all_text(app)
    assert "Something went wrong while displaying this page" in text and "abc123" not in text and not app.exception


def test_news_card_without_a_snippet_does_not_repeat_the_headline_as_a_fact():
    from models.report import NewsHighlight

    base = dict(title="Headline", url="https://x.example/n", publisher="Wire", topic="General coverage", relevance="low",
                relevance_reason="General coverage of the company with no clear link to this role.", evidence_id="E1", tier=3)
    without = c.news_card(NewsHighlight(summary="", **base))
    with_snippet = c.news_card(NewsHighlight(summary="A snippet.", **base))
    assert "Fact" not in without and "Interpretation" in without
    assert "Fact" in with_snippet and "A snippet." in with_snippet
