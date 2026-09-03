"""
LLM-powered executive report generator.

Synthesizes pipeline run metrics, ML model performance, and market
summary data into a short, business-facing report - the same
Groq-powered "intelligence engine" pattern used elsewhere in this app
(see gpt_functions.py), applied to pipeline observability instead of
building layouts. Groq's llama-3.3-70b-versatile is used in place of
GPT-4 (see groq_client.py); the report's job - turning pipeline metrics
into an executive narrative - is the same regardless of provider.

Falls back to a deterministic, non-LLM report (built straight from the
numbers) if no Groq API key is configured or the API call fails, so the
report view of the app never breaks even when GROQ_API_KEY isn't set.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import requests

from groq_client import GROQ_API_URL, GROQ_MODEL, get_groq_api_key

SYSTEM_PROMPT = (
    "You are an AI engineering intelligence analyst. You write short, "
    "sharp executive summaries for C-suite stakeholders (VP of "
    "Development, Head of Acquisitions) about a real-estate parcel "
    "ranking data pipeline. Use plain business language, not engineering "
    "jargon. Ground every claim in the numbers given - never invent "
    "figures. Respond in markdown."
)


def build_report_payload(
    pipeline_metrics: dict[str, Any],
    ml_metrics: dict[str, Any],
    market_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    """Condense raw pipeline/ML/market stats into a compact payload for
    the LLM prompt - small enough to stay well within context, and
    pre-aggregated so the model reasons over real, already-computed
    numbers rather than raw rows.
    """
    top_zones = sorted(
        market_summary, key=lambda z: z.get("high_suitability_count", 0), reverse=True
    )[:5]
    return {
        "pipeline": {
            "rows_ingested": pipeline_metrics.get("bronze_rows_ingested"),
            "rows_valid": pipeline_metrics.get("silver_rows_valid"),
            "rows_quarantined": pipeline_metrics.get("silver_rows_quarantined"),
            "rows_deduplicated": pipeline_metrics.get("silver_rows_deduped"),
            "quality_pass_rate": pipeline_metrics.get("quality_pass_rate"),
            "run_duration_seconds": pipeline_metrics.get("duration_seconds"),
        },
        "model": {
            "selected_model": ml_metrics.get("selected_model"),
            "test_accuracy": ml_metrics.get("test_accuracy"),
            "candidates_evaluated": list((ml_metrics.get("candidates_evaluated") or {}).keys()),
            "n_train_rows": ml_metrics.get("n_train_rows"),
            "n_test_rows": ml_metrics.get("n_test_rows"),
        },
        "top_zoning_districts": top_zones,
    }


def _fallback_report(payload: dict[str, Any]) -> str:
    """Deterministic, numbers-only report used when no Groq key is
    configured or the API call fails - keeps the report view usable
    without a live LLM call.
    """
    p, m = payload["pipeline"], payload["model"]
    lines = [
        "## Pipeline Executive Summary (offline fallback)",
        "",
        f"- Ingested **{p['rows_ingested']:,}** parcel/zoning records this run; "
        f"**{p['quality_pass_rate']:.1%}** passed quality validation "
        f"({p['rows_quarantined']:,} quarantined, {p['rows_deduplicated']:,} deduplicated).",
        f"- Site-ranking model **{m['selected_model']}** achieved "
        f"**{m['test_accuracy']:.1%}** accuracy on held-out data "
        f"({m['n_test_rows']:,} test parcels), selected from "
        f"{len(m['candidates_evaluated'])} candidate model families.",
        f"- Pipeline completed in {p['run_duration_seconds']}s end-to-end.",
        "",
        "_This is a deterministic fallback report generated directly from "
        "pipeline metrics. Configure GROQ_API_KEY to enable the "
        "LLM-generated narrative version._",
    ]
    return "\n".join(lines)


def generate_executive_report(payload: dict[str, Any], api_key: Optional[str] = None) -> str:
    """Call Groq to turn `payload` into an executive-ready markdown
    report. Falls back to a deterministic numbers-only report if no key
    is configured or the call fails for any reason - this must never
    raise, since it backs a UI report view.
    """
    key = api_key or get_groq_api_key()
    if not key:
        return _fallback_report(payload)

    prompt = f"""
    Here are this run's pipeline, model, and market metrics as JSON:

    {json.dumps(payload, indent=2, default=str)}

    Write a concise executive report (roughly 200-300 words) with:
    1. A one-paragraph headline summary of pipeline health and data quality.
    2. A short section on the site-ranking model's performance and what
       it means for parcel-selection confidence.
    3. A short section on market/zoning trends worth flagging to
       leadership, grounded only in the top_zoning_districts data given.
    Use markdown headings. Do not invent numbers not present above.
    """

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=30)
        result = response.json()
        if "choices" not in result:
            return _fallback_report(payload)
        return result["choices"][0]["message"]["content"]
    except Exception:
        return _fallback_report(payload)


if __name__ == "__main__":
    from pathlib import Path

    import pandas as pd

    pipeline_metrics = json.loads(Path("data/pipeline_status.json").read_text())
    ml_metrics = json.loads(Path("models/latest_metrics.json").read_text())
    market_summary = pd.read_parquet("data/gold/market_summary_gold.parquet").to_dict(orient="records")

    payload = build_report_payload(pipeline_metrics, ml_metrics, market_summary)
    print(generate_executive_report(payload))
