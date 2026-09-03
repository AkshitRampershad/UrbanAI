"""
Bronze -> Silver -> Gold medallion pipeline for parcel/zoning data.

Runs entirely on local Parquet files via DuckDB, standing in for a
Databricks Lakehouse (Auto Loader for incremental file discovery, DLT
for governed ingestion/schema evolution/quality checks, LakeFlow for
orchestration). See pipeline/cloud_adapters.py for where the real
Databricks/AWS calls would plug in when that infrastructure is
available - the interfaces here are shaped to match.

Bronze: raw landing, append-only, source data untouched aside from
        ingestion metadata (mirrors Auto Loader's incremental file
        discovery + schema inference).
Silver: validated/cleaned - deduped, nulls handled, out-of-range values
        quarantined (mirrors DLT expectations).
Gold:   feature-engineered, aggregated tables ready for ML training and
        BI/reporting.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
GOLD_DIR = Path("data/gold")

for _dir in (BRONZE_DIR, SILVER_DIR, GOLD_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineRunMetrics:
    run_id: str
    bronze_rows_ingested: int = 0
    silver_rows_valid: int = 0
    silver_rows_quarantined: int = 0
    silver_rows_deduped: int = 0
    gold_rows: int = 0
    quality_pass_rate: float = 0.0
    duration_seconds: float = 0.0
    zoning_summary: list = field(default_factory=list)


def bronze_ingest(source_csv: str, batch_id: str | None = None) -> tuple[Path, int]:
    """Land raw source data into the Bronze layer, tagging each row with
    ingestion metadata. Mirrors Auto Loader's incremental file discovery:
    each call is a new batch.

    Re-landing the *same* source file (e.g. re-running the demo pipeline
    against the same sample CSV) replaces that file's prior batches rather
    than piling up duplicate full-dataset copies in Bronze - a real Auto
    Loader would only ever see genuinely new files, so this keeps repeated
    demo runs honest instead of making Silver's dedup step absorb the same
    25k rows over and over. Batches landed from a different source (e.g.
    the /ingest API) are untouched.
    """
    batch_id = batch_id or uuid.uuid4().hex[:8]
    source_name = Path(source_csv).name

    for stale in BRONZE_DIR.glob("batch_*.parquet"):
        try:
            if pd.read_parquet(stale, columns=["_source_file"])["_source_file"].iloc[0] == source_name:
                stale.unlink()
        except Exception:
            pass

    con = duckdb.connect()
    df = con.execute(f"""
        SELECT *,
               '{batch_id}' AS _ingest_batch_id,
               now() AS _ingested_at,
               '{source_name}' AS _source_file
        FROM read_csv_auto('{source_csv}')
    """).df()
    con.close()

    out_path = BRONZE_DIR / f"batch_{batch_id}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path, len(df)


def silver_clean(bronze_glob: str = "data/bronze/*.parquet") -> dict:
    """Validate and clean Bronze data into Silver: dedupe on parcel_id,
    quarantine rows that fail schema/range checks (mirrors DLT
    expectations), and write both the valid Silver table and a
    quarantine table for auditability.
    """
    con = duckdb.connect()
    raw = con.execute(f"SELECT * FROM read_parquet('{bronze_glob}')").df()

    total_in = len(raw)

    # Quality expectations, mirroring DLT-style declarative constraints.
    valid_mask = (
        raw["lot_size_acres"].notna()
        & (raw["lot_size_acres"] > 0)
        & raw["assessed_value"].notna()
        & (raw["assessed_value"] > 0)
        & raw["latitude"].between(36.0, 40.0)
        & raw["longitude"].between(-80.0, -75.0)
    )

    quarantined = raw[~valid_mask].copy()
    valid = raw[valid_mask].copy()

    before_dedup = len(valid)
    valid = valid.sort_values("_ingested_at").drop_duplicates(subset="parcel_id", keep="last")
    deduped_count = before_dedup - len(valid)

    valid.to_parquet(SILVER_DIR / "parcels_silver.parquet", index=False)
    quarantined.to_parquet(SILVER_DIR / "parcels_quarantine.parquet", index=False)
    con.close()

    return {
        "rows_in": total_in,
        "rows_valid": len(valid),
        "rows_quarantined": len(quarantined),
        "rows_deduped": deduped_count,
        "quality_pass_rate": round(len(valid) / total_in, 4) if total_in else 0.0,
    }


def gold_aggregate() -> dict:
    """Build Gold-layer feature and aggregate tables from Silver:
    - parcels_gold.parquet: ML-ready feature table
    - market_summary_gold.parquet: zoning-level aggregates for reporting
    """
    con = duckdb.connect()
    silver_path = SILVER_DIR / "parcels_silver.parquet"

    features = con.execute(f"""
        SELECT
            parcel_id, latitude, longitude, zoning_code, land_use_current,
            lot_size_acres, distance_to_highway_mi, distance_to_transit_mi,
            population_density_per_sqmi, median_income_nearby,
            utilities_available, floodplain_flag, last_sale_year,
            assessed_value, development_suitability
        FROM read_parquet('{silver_path}')
    """).df()
    features.to_parquet(GOLD_DIR / "parcels_gold.parquet", index=False)

    summary = con.execute(f"""
        SELECT
            zoning_code,
            COUNT(*) AS parcel_count,
            ROUND(AVG(assessed_value), 0) AS avg_assessed_value,
            ROUND(AVG(lot_size_acres), 2) AS avg_lot_size_acres,
            SUM(CASE WHEN development_suitability = 'High' THEN 1 ELSE 0 END) AS high_suitability_count
        FROM read_parquet('{silver_path}')
        GROUP BY zoning_code
        ORDER BY high_suitability_count DESC
    """).df()
    summary.to_parquet(GOLD_DIR / "market_summary_gold.parquet", index=False)
    con.close()

    return {
        "gold_rows": len(features),
        "zoning_summary": summary.to_dict(orient="records"),
    }


def run_pipeline(source_csv: str = "data/raw/parcels_sample.csv") -> PipelineRunMetrics:
    """Run the full Bronze -> Silver -> Gold pipeline once and return
    run metrics for the executive report / UI.
    """
    start = time.time()
    run_id = uuid.uuid4().hex[:8]

    _, bronze_rows = bronze_ingest(source_csv, batch_id=run_id)
    silver_stats = silver_clean()
    gold_stats = gold_aggregate()

    return PipelineRunMetrics(
        run_id=run_id,
        bronze_rows_ingested=bronze_rows,
        silver_rows_valid=silver_stats["rows_valid"],
        silver_rows_quarantined=silver_stats["rows_quarantined"],
        silver_rows_deduped=silver_stats["rows_deduped"],
        gold_rows=gold_stats["gold_rows"],
        quality_pass_rate=silver_stats["quality_pass_rate"],
        duration_seconds=round(time.time() - start, 2),
        zoning_summary=gold_stats["zoning_summary"],
    )


if __name__ == "__main__":
    metrics = run_pipeline()
    print(metrics)
