"""
FastAPI data ingestion/query service for the parcel pipeline.

Run locally with:
    uvicorn api.ingestion_service:app --reload --port 8000

Endpoints:
    POST /ingest        - accept a batch of parcel records, land them in Bronze
    POST /pipeline/run   - run the full Bronze -> Silver -> Gold pipeline
    GET  /pipeline/status - latest pipeline run metrics
    GET  /parcels/top    - top-N parcels by predicted development suitability
    GET  /parcels/{id}   - look up a single parcel from the Gold layer
    GET  /health          - liveness check

In production this would sit in front of Databricks Auto Loader/DLT
(see pipeline/cloud_adapters.py for the AWS S3 + Databricks placeholder
wiring) rather than local Parquet files.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline.cloud_adapters import BACKEND, is_cloud_backend_enabled
from pipeline.executive_report import build_report_payload, generate_executive_report
from pipeline.generate_sample_data import generate_parcels
from pipeline.medallion import GOLD_DIR, run_pipeline

app = FastAPI(
    title="UrbanAI Parcel Ingestion Service",
    description="Ingests, validates, and serves parcel/zoning data for the TerraIQ site-ranking pipeline.",
    version="1.0.0",
)

STATUS_PATH = Path("data/pipeline_status.json")


class ParcelRecord(BaseModel):
    parcel_id: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    zoning_code: str
    land_use_current: str
    lot_size_acres: float = Field(gt=0)
    distance_to_highway_mi: float = Field(ge=0)
    distance_to_transit_mi: float = Field(ge=0)
    population_density_per_sqmi: float = Field(ge=0)
    median_income_nearby: float = Field(ge=0)
    utilities_available: bool
    floodplain_flag: bool
    last_sale_year: int
    assessed_value: float = Field(gt=0)
    development_suitability: Optional[str] = None


class IngestBatch(BaseModel):
    records: list[ParcelRecord]


class IngestResponse(BaseModel):
    batch_id: str
    rows_received: int
    file_path: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": BACKEND, "cloud_backend_enabled": is_cloud_backend_enabled()}


@app.post("/ingest", response_model=IngestResponse)
def ingest_batch(batch: IngestBatch) -> IngestResponse:
    if not batch.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    batch_id = uuid.uuid4().hex[:8]
    df = pd.DataFrame([r.model_dump() for r in batch.records])
    df["_ingest_batch_id"] = batch_id
    df["_ingested_at"] = pd.Timestamp.utcnow()
    df["_source_file"] = "api_ingest"

    out_dir = Path("data/bronze")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"batch_{batch_id}.parquet"
    df.to_parquet(out_path, index=False)

    return IngestResponse(batch_id=batch_id, rows_received=len(df), file_path=str(out_path))


@app.post("/pipeline/run")
def trigger_pipeline_run(source_csv: str = "data/raw/parcels_sample.csv") -> dict:
    source_path = Path(source_csv)
    if not source_path.exists():
        # First run on a fresh clone/deploy: no source file exists yet.
        # Generate the synthetic sample so the pipeline has something to
        # ingest, rather than failing with an unhandled 500.
        source_path.parent.mkdir(parents=True, exist_ok=True)
        generate_parcels().to_csv(source_path, index=False)

    metrics = run_pipeline(source_csv=source_csv)
    record = {
        "run_id": metrics.run_id,
        "bronze_rows_ingested": metrics.bronze_rows_ingested,
        "silver_rows_valid": metrics.silver_rows_valid,
        "silver_rows_quarantined": metrics.silver_rows_quarantined,
        "silver_rows_deduped": metrics.silver_rows_deduped,
        "gold_rows": metrics.gold_rows,
        "quality_pass_rate": metrics.quality_pass_rate,
        "duration_seconds": metrics.duration_seconds,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(record, indent=2))
    return record


@app.get("/pipeline/status")
def pipeline_status() -> dict:
    if not STATUS_PATH.exists():
        raise HTTPException(status_code=404, detail="No pipeline run has completed yet.")
    return json.loads(STATUS_PATH.read_text())


@app.get("/parcels/top")
def top_parcels(n: int = 10, zoning_code: Optional[str] = None) -> list[dict]:
    gold_path = GOLD_DIR / "parcels_gold.parquet"
    if not gold_path.exists():
        raise HTTPException(status_code=404, detail="Gold layer not built yet - run /pipeline/run first.")

    con = duckdb.connect()
    where_clause = f"WHERE zoning_code = '{zoning_code}'" if zoning_code else ""
    result = con.execute(f"""
        SELECT parcel_id, zoning_code, land_use_current, lot_size_acres,
               assessed_value, development_suitability
        FROM read_parquet('{gold_path}')
        {where_clause}
        ORDER BY CASE development_suitability
            WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END,
            assessed_value DESC
        LIMIT {n}
    """).df()
    con.close()
    return result.to_dict(orient="records")


@app.post("/report/generate")
def generate_report() -> dict:
    """Synthesize the latest pipeline run + model metrics + market
    summary into an executive report. Requires /pipeline/run to have
    completed at least once (needs pipeline_status.json, the trained
    model's latest_metrics.json, and the Gold market summary table).
    """
    if not STATUS_PATH.exists():
        raise HTTPException(status_code=404, detail="No pipeline run has completed yet - call /pipeline/run first.")

    metrics_path = Path("models/latest_metrics.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="No trained model yet - run pipeline/train_model.py first.")

    summary_path = GOLD_DIR / "market_summary_gold.parquet"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Gold layer not built yet - run /pipeline/run first.")

    pipeline_metrics = json.loads(STATUS_PATH.read_text())
    ml_metrics = json.loads(metrics_path.read_text())
    market_summary = pd.read_parquet(summary_path).to_dict(orient="records")

    payload = build_report_payload(pipeline_metrics, ml_metrics, market_summary)
    report_markdown = generate_executive_report(payload)
    return {"report_markdown": report_markdown, "payload": payload}


@app.get("/parcels/{parcel_id}")
def get_parcel(parcel_id: str) -> dict:
    gold_path = GOLD_DIR / "parcels_gold.parquet"
    if not gold_path.exists():
        raise HTTPException(status_code=404, detail="Gold layer not built yet - run /pipeline/run first.")

    con = duckdb.connect()
    result = con.execute(
        f"SELECT * FROM read_parquet('{gold_path}') WHERE parcel_id = ?", [parcel_id]
    ).df()
    con.close()

    if result.empty:
        raise HTTPException(status_code=404, detail=f"Parcel {parcel_id} not found.")
    return result.iloc[0].to_dict()
