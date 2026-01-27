from __future__ import annotations

import uuid
from typing import Any

import mlflow
import mlflow.sklearn # Stopped used pyfunc internals
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi import Header, HTTPException

from prometheus.serving.inference_store import ensure_store, log_inference, log_feedback
from prometheus.utils.config import get_settings


# Pydantic request and response
class PaymentFeatures(BaseModel):
    amount_gbp: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)
    is_weekend: int = Field(..., ge=0, le=1)

    merchant_category: int = Field(..., ge=0, le=10)
    payment_method: int = Field(..., ge=0, le=10)

    account_age_days: int = Field(..., ge=0)
    txn_count_24h: int = Field(..., ge=0)

    ip_country_mismatch: int = Field(..., ge=0, le=1)
    device_new: int = Field(..., ge=0, le=1)
    device_risk: float = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    proba: float
    pred: int


class FeedbackRequest(BaseModel):
    request_id: str
    outcome_label: int = Field(..., ge=0, le=1)


# app 
app = FastAPI(title="Prometheus API", version="0.1.0")

settings = get_settings()

MODEL_NAME = settings.mlflow_model_name
_model: Any | None = None
_model_version: str | None = None


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
   
    # Match training features exactly
    df = df.copy()

    df["amount_per_txn_24h"] = df["amount_gbp"] / (1.0 + df["txn_count_24h"])
    df["night_and_mismatch"] = ((df["hour"] <= 5) & (df["ip_country_mismatch"] == 1)).astype(int)
    df["new_device_and_high_velocity"] = ((df["device_new"] == 1) & (df["txn_count_24h"] >= 3)).astype(int)
    df["velocity_log"] = np.log1p(df["txn_count_24h"])

    return df


def _load_latest_model() -> tuple[Any, str]:
    """
    Load latest model version from MLflow Model Registry as a sklearn model,
    so we can call predict_proba more reliably.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    latest = client.get_latest_versions(MODEL_NAME)
    if not latest:
        raise RuntimeError(
            f"No registered versions found for model '{MODEL_NAME}'. "
            "Run training first to register a model."
        )

    v = max(latest, key=lambda mv: int(mv.version))
    model_uri = f"models:/{MODEL_NAME}/{v.version}"

    model = mlflow.sklearn.load_model(model_uri)
    return model, str(v.version)


def _reload() -> dict[str, str]:
    global _model, _model_version
    _model, _model_version = _load_latest_model()
    return {"status": "ok", "model_name": MODEL_NAME, "model_version": _model_version}


@app.on_event("startup")
def _startup() -> None:
    global _model, _model_version

    # Ensure inference store exists
    ensure_store(settings.duckdb_path)

    # Load model from MLflow
    _model, _model_version = _load_latest_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> dict[str, str]:
    return {
        "model_name": MODEL_NAME,
        "model_version": _model_version or "unknown",
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PaymentFeatures) -> PredictResponse:
    if _model is None or _model_version is None:
        raise RuntimeError("Model not loaded")

    request_id = str(uuid.uuid4())

    # Convert request to DataFrame
    features = req.model_dump()
    df = pd.DataFrame([features])

    # Apply same feature engineering as training
    df = _feature_engineering(df)

    proba = float(_model.predict_proba(df)[:, 1][0])
    pred = int(proba >= 0.5)

    # Log inference
    log_inference(
        path=settings.duckdb_path,
        request_id=request_id,
        model_name=MODEL_NAME,
        model_version=_model_version,
        features=features,
        proba=proba,
        pred=pred,
    )

    return PredictResponse(
        request_id=request_id,
        model_name=MODEL_NAME,
        model_version=_model_version,
        proba=proba,
        pred=pred,
    )

@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, str]:
    if _model_version is None:
        raise RuntimeError("Model not loaded")

    log_feedback(
        path=settings.duckdb_path,
        request_id=req.request_id,
        outcome_label=req.outcome_label,
    )
    return {"status": "ok"}


@app.post("/reload_model")
def reload_model(x_reload_token: str | None = Header(default=None)) -> dict[str, str]:
    if x_reload_token != settings.reload_token:
        raise HTTPException(status_code=401, detail="invalid reload token")
    return _reload()

