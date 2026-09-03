"""
AWS / Databricks placeholder adapters.

This project's actual pipeline runs entirely locally (DuckDB + Parquet +
scikit-learn - see pipeline/medallion.py and pipeline/train_model.py) so
it works with zero cloud credentials on Streamlit Community Cloud's free
tier. This module is the seam where a real AWS + Databricks backend would
plug in without changing any calling code in api/ingestion_service.py or
the Streamlit pages: set PIPELINE_BACKEND=databricks and populate
CloudConfig once real workspace access/credentials are available.

None of the cloud branches below are runnable as-is - they document the
exact SDK calls and config needed (S3 bucket, Databricks host/token,
AutoML experiment id, DLT pipeline id, LakeFlow job id, MLflow tracking
URI) so wiring in the real thing later is a fill-in-the-blanks exercise,
not a redesign. Each function's local equivalent is named in its
docstring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Backend = Literal["local", "databricks"]

BACKEND: Backend = "databricks" if os.environ.get("PIPELINE_BACKEND") == "databricks" else "local"


@dataclass
class CloudConfig:
    """Connection details for a real AWS + Databricks backend. Populate
    these via environment variables or Streamlit secrets before setting
    PIPELINE_BACKEND=databricks - none of them are required for the
    local backend this app runs on by default.
    """

    aws_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", ""))
    s3_bucket: str = field(default_factory=lambda: os.environ.get("S3_BUCKET", ""))
    databricks_host: str = field(default_factory=lambda: os.environ.get("DATABRICKS_HOST", ""))
    databricks_token: str = field(default_factory=lambda: os.environ.get("DATABRICKS_TOKEN", ""))
    automl_experiment_id: str = field(default_factory=lambda: os.environ.get("AUTOML_EXPERIMENT_ID", ""))
    mlflow_tracking_uri: str = field(default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI", ""))
    dlt_pipeline_id: str = field(default_factory=lambda: os.environ.get("DLT_PIPELINE_ID", ""))
    lakeflow_job_id: str = field(default_factory=lambda: os.environ.get("LAKEFLOW_JOB_ID", ""))


def is_cloud_backend_enabled() -> bool:
    return BACKEND == "databricks"


def ingest_to_s3(local_csv_path: str, config: CloudConfig | None = None) -> str:
    """Placeholder for landing raw parcel/zoning files in S3 for
    Databricks Auto Loader to discover. Production equivalent of
    pipeline.medallion.bronze_ingest's local Parquet write.

    Real implementation:

        import boto3
        s3 = boto3.client("s3", region_name=config.aws_region)
        key = f"raw/parcels/{Path(local_csv_path).name}"
        s3.upload_file(local_csv_path, config.s3_bucket, key)
        return f"s3://{config.s3_bucket}/{key}"

    Auto Loader would then pick up new files under s3://<bucket>/raw/
    incrementally via cloudFiles, inferring and evolving schema
    automatically instead of the fixed read_csv_auto() call used locally.
    """
    raise NotImplementedError(
        "AWS S3 backend not configured. Set PIPELINE_BACKEND=databricks and "
        "populate CloudConfig (AWS_REGION, S3_BUCKET) with real credentials, "
        "then implement the boto3 upload documented above."
    )


def run_dlt_pipeline(config: CloudConfig | None = None) -> dict:
    """Placeholder for triggering a Databricks DLT pipeline. Production
    equivalent of pipeline.medallion.silver_clean + gold_aggregate, but
    with declarative @dlt.expect_or_quarantine-style quality rules
    instead of the pandas boolean-mask checks used locally.

    Real implementation:

        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient(host=config.databricks_host, token=config.databricks_token)
        run = w.pipelines.start_update(pipeline_id=config.dlt_pipeline_id)
        # poll w.pipelines.get_update(...) until state == "COMPLETED"
        return {"update_id": run.update_id}
    """
    raise NotImplementedError(
        "Databricks DLT backend not configured. Populate CloudConfig "
        "(DATABRICKS_HOST, DATABRICKS_TOKEN, DLT_PIPELINE_ID) and implement "
        "the databricks-sdk calls documented above."
    )


def run_lakeflow_orchestration(config: CloudConfig | None = None) -> dict:
    """Placeholder for orchestrating the full Bronze -> Silver -> Gold
    flow as a Databricks LakeFlow job. Production equivalent of
    pipeline.medallion.run_pipeline's local sequential calls.

    Real implementation, triggered via the Jobs API:

        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient(host=config.databricks_host, token=config.databricks_token)
        run = w.jobs.run_now(job_id=int(config.lakeflow_job_id))
        return {"run_id": run.run_id}
    """
    raise NotImplementedError(
        "Databricks LakeFlow backend not configured. Populate CloudConfig "
        "(DATABRICKS_HOST, DATABRICKS_TOKEN, LAKEFLOW_JOB_ID)."
    )


def train_with_automl(config: CloudConfig | None = None) -> dict:
    """Placeholder for Databricks AutoML. Production equivalent of
    pipeline.train_model.train_and_select_best's local candidate-model
    search across LogisticRegression/RandomForest/GradientBoosting.

    Real implementation:

        import databricks.automl as automl
        summary = automl.classify(
            dataset=spark_df,
            target_col="development_suitability",
            experiment_name=config.automl_experiment_id,
            timeout_minutes=30,
        )
        return {"best_trial_uri": summary.best_trial.model_path}

    AutoML searches a much larger space of model families and
    hyperparameters in parallel across a Databricks cluster than the 3
    candidates this project evaluates locally in seconds.
    """
    raise NotImplementedError(
        "Databricks AutoML backend not configured. Populate CloudConfig "
        "(DATABRICKS_HOST, DATABRICKS_TOKEN, AUTOML_EXPERIMENT_ID)."
    )


def log_to_mlflow(run_record: dict, config: CloudConfig | None = None) -> None:
    """Placeholder for logging a training run to a real MLflow tracking
    server. Production equivalent of the local models/mlflow_runs.json
    JSON log written by pipeline.train_model.

    Real implementation:

        import mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        with mlflow.start_run():
            mlflow.log_params({"selected_model": run_record["selected_model"]})
            mlflow.log_metrics({"test_accuracy": run_record["test_accuracy"]})
            mlflow.sklearn.log_model(run_record["model"], "model")
    """
    raise NotImplementedError(
        "MLflow tracking server not configured. Set MLFLOW_TRACKING_URI and "
        "implement the mlflow logging calls documented above."
    )
