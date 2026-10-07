"""Tests for the offline monitoring checks (PSI drift, data quality, predictions)."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data import binarize_target, load_data, make_split
from src.monitor import (
    PSI_ALERT,
    PSI_ATTENTION,
    data_quality,
    drift_report,
    prediction_summary,
    psi_categorical,
    psi_numeric,
)
from src.predict import model_features

rng = np.random.default_rng(0)


def test_psi_is_near_zero_for_the_same_distribution():
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, b) < PSI_ATTENTION


def test_psi_flags_a_shifted_distribution():
    a = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(a, a + 1.0) > PSI_ALERT


def test_psi_categorical_detects_share_change_and_new_categories():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    assert psi_categorical(ref, ref) == pytest.approx(0.0)
    assert psi_categorical(ref, pd.Series(["a"] * 90 + ["b"] * 10)) > PSI_ALERT
    assert psi_categorical(ref, pd.Series(["a"] * 50 + ["c"] * 50)) > PSI_ALERT


@pytest.fixture(scope="module")
def split():
    cfg = load_config()
    if not (cfg.model_path.exists() and cfg.raw_data_path.exists()):
        pytest.skip("Model artifact or dataset not available")
    X_train, X_test, _, _ = make_split(binarize_target(load_data(cfg), cfg), cfg.primary_variant, cfg)
    return X_train, X_test


def test_held_out_students_show_no_drift(split):
    X_train, X_test = split
    report = drift_report(X_train, X_test, model_features())
    assert set(report["feature"]) == set(model_features())
    assert (report["status"] == "ok").all()


def test_drift_report_flags_a_simulated_shift(split):
    X_train, X_test = split
    shifted = X_test.copy()
    shifted["UnidadesCurriculares1SemestreAprovado"] = 0  # nobody passes anything
    report = drift_report(X_train, shifted, model_features()).set_index("feature")
    assert report.loc["UnidadesCurriculares1SemestreAprovado", "status"] == "alerta"


def test_data_quality_reports_missing_columns_and_corrupted_grades(split):
    X_train, X_test = split
    batch = X_test.drop(columns=["Curso"]).copy()
    batch["UnidadesCurriculares1SemestreGrau"] = 13875.0  # shifted-decimal corruption
    quality = data_quality(X_train, batch, model_features())
    assert quality["missing_columns"] == ["Curso"]
    assert quality["corrupted_grade_share"]["UnidadesCurriculares1SemestreGrau"] == 1.0


def test_prediction_summary_matches_the_test_set_reference(split):
    _, X_test = split
    summary = prediction_summary(X_test)
    assert summary["mean_proba"] == pytest.approx(summary["reference_test_mean_proba"])
    assert summary["alert_rate"] == pytest.approx(summary["reference_test_alert_rate"])
    assert sum(summary["risk_share"].values()) == pytest.approx(1.0)
