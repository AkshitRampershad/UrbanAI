"""
Shared Groq API client config.

Reads the API key from Streamlit secrets when running inside a Streamlit
app, falling back to the GROQ_API_KEY environment variable otherwise, so
the FastAPI service, pipeline CLI scripts, and tests can all use the same
helper without requiring a live Streamlit runtime or a populated
secrets.toml.
"""

from __future__ import annotations

import os
from typing import Optional

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# mixtral-8x7b-32768 was decommissioned first; its replacement,
# llama-3.3-70b-versatile, was itself deprecated by Groq in June 2026.
# openai/gpt-oss-120b is Groq's current recommended replacement for
# general-purpose/agentic use, with JSON-mode support.
GROQ_MODEL = "openai/gpt-oss-120b"


def get_groq_api_key() -> Optional[str]:
    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")
