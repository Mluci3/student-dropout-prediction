"""Tests for the inference layer (risk mapping + end-to-end preprocessing)."""

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.data import binarize_target, load_data, split_xy
from src.features import build_preprocessor
from src.predict import risk_label


def test_risk_label_bands():
    # threshold 0.4: below -> Baixo, between -> Médio, well above 0.5 -> Alto
    assert risk_label(0.10, 0.4) == "Baixo"
    assert risk_label(0.45, 0.4) == "Médio"
    assert risk_label(0.90, 0.4) == "Alto"


@pytest.fixture(scope="module")
def xy():
    cfg = load_config()
    if not cfg.raw_data_path.exists():
        pytest.skip("Raw dataset not available")
    df = binarize_target(load_data(cfg), cfg)
    return split_xy(df, cfg.primary_variant)


def test_full_pipeline_fits_and_outputs_valid_probabilities(xy):
    X, y = xy
    X_small, y_small = X.head(400), y.head(400)
    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor(load_config().primary_variant)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(X_small, y_small)
    proba = pipe.predict_proba(X_small.head(10))[:, 1]
    assert proba.shape == (10,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_threshold_for_recall_meets_target():
    import numpy as np

    from src.evaluate import choose_threshold_for_recall

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    for target in (0.5, 0.8, 0.95):
        t = choose_threshold_for_recall(y, proba, target)
        recall = ((proba >= t) & (y == 1)).sum() / (y == 1).sum()
        assert recall >= target
        # and it is the strictest such threshold: any higher cutoff misses the target
        higher = proba[proba > t]
        if len(higher):
            t2 = higher.min()
            assert ((proba >= t2) & (y == 1)).sum() / (y == 1).sum() < target


def test_saved_artifact_predicts_and_explains_a_real_student():
    """Smoke test of the served model (calibrated pipeline) + its explanation."""
    from src.config import load_config
    from src.explain import explain_one, load_background
    from src.predict import load_pipeline, model_features, predict_one

    cfg = load_config()
    if not (cfg.model_path.exists() and cfg.raw_data_path.exists()):
        pytest.skip("Model artifact or dataset not available")
    student = load_data(cfg)[model_features()].iloc[[0]]
    result = predict_one(student.iloc[0].to_dict())
    assert 0.0 <= result["dropout_proba"] <= 1.0
    assert result["risk"] in {"Baixo", "Médio", "Alto"}
    factors = explain_one(load_pipeline(), student, top_n=5, background=load_background())
    assert len(factors) == 5
    assert any(abs(f["shap_value"]) > 0 for f in factors)
