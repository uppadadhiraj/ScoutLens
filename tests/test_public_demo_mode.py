"""Hosted showcase mode (PUBLIC_DEMO=1): only the demo runs, nothing is stored, no live inputs are offered."""
from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
from services.cache.repository import InvestigationRepository

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning]
    parts += [s.value for s in at.success] + [i.value for i in at.info]
    return "\n".join(parts)


def _click(at, label):
    next(b for b in at.button if b.label == label).click().run()


@pytest.fixture
def public_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "public.db"))
    monkeypatch.setenv("SERPAPI_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("PUBLIC_DEMO", "1")
    config.get_settings.cache_clear()
    st.cache_resource.clear()
    yield AppTest.from_file(APP, default_timeout=90).run()
    config.get_settings.cache_clear()
    st.cache_resource.clear()


def test_public_demo_is_off_by_default():
    assert config.Settings(_env_file=None).public_demo is False
    assert config.Settings(_env_file=None).public_status()["public_demo"] is False


def test_public_demo_flag_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("PUBLIC_DEMO", "true")
    assert config.Settings(_env_file=None).public_demo is True


def test_home_offers_only_the_demo(public_app):
    assert not public_app.exception
    assert "Public demo" in _text(public_app)
    assert [b.label for b in public_app.button] == ["Try the demo"]
    assert len(public_app.text_input) == 0  # no URL / company / preference fields
    assert len(public_app.text_area) == 0  # no pasted job description


def test_demo_still_runs_end_to_end_but_nothing_is_saved(public_app, tmp_path):
    _click(public_app, "Try the demo")
    assert not public_app.exception
    assert "Investigation complete" in _text(public_app)
    assert InvestigationRepository(tmp_path / "public.db").list() == []


def test_history_page_is_closed_so_visitors_cannot_see_each_other(public_app):
    _click(public_app, "Try the demo")
    public_app.session_state["view"] = "history"
    public_app.run()
    assert not public_app.exception
    text = _text(public_app)
    assert "History is turned off in the public demo" in text
    assert "Software Engineer Intern" not in text
    assert [b.label for b in public_app.button if b.label in ("Open", "Delete")] == []
