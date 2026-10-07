"""Inference helpers shared by the app and any batch scoring.

Loads the serialized pipeline once and exposes single-record and batch
prediction functions that return dropout probabilities and the risk label
implied by the tuned decision threshold.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from .config import feature_columns, load_config


@lru_cache(maxsize=1)
def load_pipeline():
    """Load and cache the fitted pipeline from ``models/pipeline.joblib``."""
    cfg = load_config()
    if not cfg.model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {cfg.model_path}. "
            "Rode `python -m src.train` primeiro."
        )
    return joblib.load(cfg.model_path)


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    cfg = load_config()
    if not cfg.metrics_path.exists():
        return {}
    with open(cfg.metrics_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def served_variant() -> str:
    """Variant of the serialized model (written by training next to the artifact)."""
    return load_metrics().get("variant", load_config().primary_variant)


def model_features() -> list[str]:
    """Raw input columns expected by the serialized model."""
    return load_metrics().get("feature_columns") or feature_columns(served_variant())


def get_threshold() -> float:
    return float(load_metrics().get("threshold", load_config().train["default_threshold"]))


def risk_label(proba: float, threshold: float) -> str:
    """Map a calibrated probability to a three-level risk band.

    - Baixo: below the alert threshold (no alert);
    - Médio: alert raised, but dropout is still less likely than not;
    - Alto: dropout is more likely than not (calibrated proba >= 50%).
    """
    if proba >= max(threshold, 0.5):
        return "Alto"
    if proba >= threshold:
        return "Médio"
    return "Baixo"


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of raw feature rows; returns proba, prediction, risk."""
    pipeline = load_pipeline()
    threshold = get_threshold()
    X = df[model_features()].copy()
    proba = pipeline.predict_proba(X)[:, 1]
    out = df.copy()
    out["dropout_proba"] = proba
    out["dropout_pred"] = (proba >= threshold).astype(int)
    out["risk"] = [risk_label(p, threshold) for p in proba]
    return out


def predict_one(record: dict) -> dict:
    """Score a single student given a dict of raw feature values."""
    scored = predict_batch(pd.DataFrame([record]))
    return {
        "dropout_proba": float(scored["dropout_proba"].iloc[0]),
        "dropout_pred": int(scored["dropout_pred"].iloc[0]),
        "risk": str(scored["risk"].iloc[0]),
        "threshold": get_threshold(),
    }
