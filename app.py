"""ScoutLens entry point: `streamlit run app.py`."""
from __future__ import annotations

import streamlit as st

from config import get_settings
from utils.logging_setup import setup_logging

settings = get_settings()
setup_logging(settings.log_level)

st.set_page_config(page_title="ScoutLens", page_icon="🔎", layout="wide")

st.title("ScoutLens")
st.caption("Investigate the opportunity before you invest your time.")

status = settings.public_status()
col_a, col_b = st.columns(2)
col_a.metric("SerpApi key", "configured" if status["serpapi_configured"] else "missing")
col_b.metric("LLM provider", f"{status['llm_provider']} ({status['llm_model'] or 'n/a'})")

if not status["serpapi_configured"]:
    st.warning("Set SERPAPI_KEY in your .env file to run live investigations. See .env.example.")
