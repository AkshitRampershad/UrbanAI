"""
Synthetic parcel/zoning sample data generator.

NOTE ON DATA PROVENANCE: This generates SYNTHETIC data with realistic
field names, value ranges, and correlations modeled on public Loudoun
County, VA parcel/zoning records. It is NOT scraped or downloaded from
any live government system - it exists so the rest of this pipeline
(medallion ETL, ML ranking model, FastAPI service) has something real
to ingest, clean, and train on end-to-end. Swap this module out for a
real bulk parcel/zoning export (e.g. a county GIS open-data download)
to run the same pipeline on real data.

The production version of this project (Surge Infotech) ingested 50M+
real parcel/zoning records via Databricks Auto Loader into S3; this
generator produces a scaled-down sample (tens of thousands of rows) so
the full pipeline runs end-to-end on a laptop or Streamlit Cloud's free
tier in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Loudoun County, VA approximate bounding box
LAT_RANGE = (38.85, 39.35)
LON_RANGE = (-77.95, -77.30)

ZONING_CODES = ["R1", "R2", "R4", "R8", "PD-H", "PD-CB", "TR-10", "TR-3", "JLMA", "AR-1"]
LAND_USE = ["Vacant", "Single-Family", "Multi-Family", "Commercial", "Agricultural", "Industrial"]


def generate_parcels(n_rows: int = 25_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic parcel/zoning dataset with realistic, learnable
    structure so a downstream ML model has real signal to learn from.
    """
    rng = np.random.default_rng(seed)

    lat = rng.uniform(*LAT_RANGE, n_rows)
    lon = rng.uniform(*LON_RANGE, n_rows)
    zoning_code = rng.choice(ZONING_CODES, n_rows)
    land_use_current = rng.choice(LAND_USE, n_rows, p=[0.18, 0.35, 0.12, 0.15, 0.12, 0.08])
    lot_size_acres = np.round(rng.lognormal(mean=0.2, sigma=0.9, size=n_rows).clip(0.05, 50), 2)
    distance_to_highway_mi = np.round(rng.exponential(scale=2.5, size=n_rows).clip(0.05, 25), 2)
    distance_to_transit_mi = np.round(rng.exponential(scale=4.0, size=n_rows).clip(0.05, 30), 2)
    population_density = np.round(rng.gamma(shape=2.0, scale=800, size=n_rows), 0)
    median_income_nearby = np.round(rng.normal(loc=125_000, scale=35_000, size=n_rows).clip(35_000, 350_000), 0)
    utilities_available = rng.choice([True, False], n_rows, p=[0.82, 0.18])
    floodplain_flag = rng.choice([True, False], n_rows, p=[0.09, 0.91])
    last_sale_year = rng.integers(2005, 2025, n_rows)
    assessed_value = np.round(
        (lot_size_acres * rng.uniform(40_000, 120_000, n_rows))
        + (median_income_nearby * 0.6)
        - (distance_to_highway_mi * 8_000)
        + rng.normal(0, 25_000, n_rows),
        0,
    ).clip(15_000, None)

    # Latent "development suitability" signal: a realistic composite of the
    # factors a real site-ranking model would weigh, plus noise so it's
    # learnable but not trivially deterministic from any single feature.
    raw_score = (
        0.30 * (1 / (1 + distance_to_highway_mi))
        + 0.20 * (1 / (1 + distance_to_transit_mi))
        + 0.15 * (population_density / population_density.max())
        + 0.15 * utilities_available.astype(float)
        + 0.10 * (1 - floodplain_flag.astype(float))
        + 0.10 * (median_income_nearby / median_income_nearby.max())
        + rng.normal(0, 0.07, n_rows)
    )
    suitability_label = pd.qcut(raw_score, q=[0, 0.6, 0.85, 1.0], labels=["Low", "Medium", "High"])

    df = pd.DataFrame({
        "parcel_id": [f"LC-{i:07d}" for i in range(n_rows)],
        "latitude": lat,
        "longitude": lon,
        "zoning_code": zoning_code,
        "land_use_current": land_use_current,
        "lot_size_acres": lot_size_acres,
        "distance_to_highway_mi": distance_to_highway_mi,
        "distance_to_transit_mi": distance_to_transit_mi,
        "population_density_per_sqmi": population_density,
        "median_income_nearby": median_income_nearby,
        "utilities_available": utilities_available,
        "floodplain_flag": floodplain_flag,
        "last_sale_year": last_sale_year,
        "assessed_value": assessed_value,
        "development_suitability": suitability_label.astype(str),
    })

    # Inject realistic data-quality issues for the Silver-layer quality
    # checks to actually catch and fix (nulls, duplicates, out-of-range).
    dirty_idx = rng.choice(n_rows, size=int(n_rows * 0.015), replace=False)
    df.loc[dirty_idx[: len(dirty_idx) // 3], "assessed_value"] = np.nan
    df.loc[dirty_idx[len(dirty_idx) // 3: 2 * len(dirty_idx) // 3], "lot_size_acres"] = -1
    dup_rows = df.sample(n=int(n_rows * 0.005), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=25_000)
    parser.add_argument("--out", type=str, default="data/raw/parcels_sample.csv")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = generate_parcels(n_rows=args.rows)
    data.to_csv(args.out, index=False)
    print(f"Wrote {len(data):,} synthetic parcel records to {args.out}")
