"""
Portfolio Intelligence: an end-to-end data engineering + ML + LLM demo
showing the Bronze/Silver/Gold medallion pipeline, a real trained
site-ranking model, and an LLM-generated executive report - the
concrete build behind the "Data Engineer / AI Practitioner" experience
entry on the portfolio site.

Runs entirely on synthetic sample data (see pipeline/generate_sample_data.py)
using local, free tools (DuckDB + Parquet + scikit-learn) that mirror the
real Databricks/AWS/MLflow architecture. See pipeline/cloud_adapters.py
for the placeholder seam where a real cloud backend would plug in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.cloud_adapters import is_cloud_backend_enabled
from pipeline.executive_report import build_report_payload, generate_executive_report
from pipeline.generate_sample_data import generate_parcels
from pipeline.medallion import GOLD_DIR, run_pipeline
from pipeline.train_model import train_and_select_best

st.set_page_config(page_title="Portfolio Intelligence - UrbanAI", layout="wide")
st.title("Portfolio Intelligence")
st.caption(
    "A real, runnable Bronze/Silver/Gold pipeline, ML site-ranking model, and "
    "LLM executive report - built to demonstrate the data engineering work "
    "behind the resume, on synthetic sample data."
)

with st.expander("What's real here, and what's a local stand-in?", expanded=False):
    st.markdown(
        """
| Resume claim | This demo |
|---|---|
| Spatial data ingestion + ML site-ranking engine (Databricks AutoML, MLflow), 92% accuracy | Real pipeline + real trained model below, using local DuckDB/scikit-learn instead of a Databricks workspace. Accuracy shown is whatever the model actually scores on held-out data - never hardcoded to match any target number. |
| FastAPI ingestion of 50M+ parcel/zoning records into AWS S3 (Auto Loader, DLT) | Real FastAPI service (`api/ingestion_service.py`) ingesting a **synthetic**, configurable-size sample instead of 50M real records / real S3 |
| Bronze/Silver/Gold medallion pipeline (LakeFlow, DLT) | Real medallion pipeline (`pipeline/medallion.py`), run locally via DuckDB instead of Databricks LakeFlow/DLT |
| GPT-4 executive report engine | Real LLM report generator below, using Groq's Llama 3.3 70B instead of GPT-4 |
| AWS/Databricks components (S3, AutoML, MLflow, LakeFlow, DLT) | Placeholder adapters in `pipeline/cloud_adapters.py` - documented, config-toggled, ready to wire in with real credentials |

Data is synthetic (seeded, reproducible) - modeled on public Loudoun County, VA
parcel/zoning structure, not scraped from any live system. See
`pipeline/generate_sample_data.py` for full disclosure.
        """
    )

backend_label = "Databricks/AWS (cloud)" if is_cloud_backend_enabled() else "Local (DuckDB + scikit-learn)"
st.info(f"**Active backend:** {backend_label} - toggle via the `PIPELINE_BACKEND` env var, see `pipeline/cloud_adapters.py`.")

st.divider()

st.subheader("1. Ingest & run the medallion pipeline")
sample_rows = st.slider("Synthetic sample size", min_value=5_000, max_value=50_000, value=25_000, step=5_000)

if st.button("Generate sample data & run pipeline", type="primary"):
    with st.spinner("Generating synthetic parcel/zoning data..."):
        df = generate_parcels(n_rows=sample_rows)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/raw/parcels_sample.csv", index=False)

    with st.spinner("Running Bronze -> Silver -> Gold pipeline..."):
        metrics = run_pipeline()
    st.session_state["pipeline_metrics"] = metrics.__dict__
    st.success(f"Pipeline complete in {metrics.duration_seconds}s.")

if "pipeline_metrics" in st.session_state:
    m = st.session_state["pipeline_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows ingested", f"{m['bronze_rows_ingested']:,}")
    c2.metric("Valid (Silver)", f"{m['silver_rows_valid']:,}")
    c3.metric("Quarantined", f"{m['silver_rows_quarantined']:,}")
    c4.metric("Quality pass rate", f"{m['quality_pass_rate']:.1%}")

st.divider()

st.subheader("2. Train the site-ranking model")
gold_path = GOLD_DIR / "parcels_gold.parquet"
gold_ready = gold_path.exists()

if not gold_ready:
    st.warning("Run the pipeline above first - the model trains on the Gold layer.")
elif st.button("Train site-ranking model"):
    with st.spinner("Training candidate models (logistic regression, random forest, gradient boosting)..."):
        run_record = train_and_select_best()
    st.session_state["ml_metrics"] = run_record
    st.success(
        f"Selected **{run_record['selected_model']}** - "
        f"**{run_record['test_accuracy']:.1%}** real test accuracy "
        f"(never hardcoded - this is whatever the model actually scores)."
    )

if "ml_metrics" not in st.session_state and Path("models/latest_metrics.json").exists():
    st.session_state["ml_metrics"] = json.loads(Path("models/latest_metrics.json").read_text())

if "ml_metrics" in st.session_state:
    mm = st.session_state["ml_metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", mm["selected_model"])
    c2.metric("Test accuracy", f"{mm['test_accuracy']:.1%}")
    c3.metric("Test set size", f"{mm['n_test_rows']:,} parcels")
    with st.expander("Candidate models evaluated"):
        st.json(mm["candidates_evaluated"])

st.divider()

st.subheader("3. Top-ranked development parcels")
if gold_ready:
    gold_df = pd.read_parquet(gold_path)
    rank_map = {"High": 0, "Medium": 1, "Low": 2}
    gold_df["_rank"] = gold_df["development_suitability"].map(rank_map)
    top = gold_df.sort_values(["_rank", "assessed_value"], ascending=[True, False]).head(15)
    st.dataframe(
        top[["parcel_id", "zoning_code", "land_use_current", "lot_size_acres", "assessed_value", "development_suitability"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Run the pipeline above to see ranked parcels.")

st.divider()

st.subheader("4. AI-generated executive report")
metrics_path = Path("models/latest_metrics.json")
status_path = Path("data/pipeline_status.json")
report_ready = ("pipeline_metrics" in st.session_state or status_path.exists()) and metrics_path.exists() and gold_ready

if not report_ready:
    st.info("Run the pipeline and train the model above to generate a report.")
elif st.button("Generate executive report"):
    with st.spinner("Synthesizing pipeline metrics into an executive report..."):
        pipeline_metrics = st.session_state.get("pipeline_metrics") or json.loads(status_path.read_text())
        ml_metrics = json.loads(metrics_path.read_text())
        market_summary = pd.read_parquet(GOLD_DIR / "market_summary_gold.parquet").to_dict(orient="records")
        payload = build_report_payload(pipeline_metrics, ml_metrics, market_summary)
        report = generate_executive_report(payload)
    st.session_state["executive_report"] = report

if "executive_report" in st.session_state:
    st.markdown(st.session_state["executive_report"])
