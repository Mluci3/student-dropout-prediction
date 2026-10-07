"""Tests for target binarization, feature sets and the stratified split."""

import pandas as pd
import pytest

from src.config import SENSITIVE_COLUMNS, VARIANT_SEMESTERS, feature_columns, load_config
from src.data import TARGET, binarize_target, load_data, make_split


def test_binarize_target_only_dropout_is_positive():
    df = pd.DataFrame({"Target": ["Desistente", "Graduado", "Matriculado", "Desistente"]})
    out = binarize_target(df)
    assert out[TARGET].tolist() == [1, 0, 0, 1]


def test_early_warning_features_exclude_second_semester():
    cols = feature_columns("early_warning")
    assert not [c for c in cols if "2Semestre" in c]
    assert set(cols) < set(feature_columns("full"))


@pytest.mark.parametrize("variant", list(VARIANT_SEMESTERS))
def test_sensitive_attributes_are_not_model_features(variant):
    # ADR 0007: gender, nationality and marital status never reach the model.
    assert not set(SENSITIVE_COLUMNS) & set(feature_columns(variant))


def test_primary_variant_is_early_warning():
    # Documentation, app and ADR 0006 assume the served model predicts at the
    # end of the 1st semester; changing this must be a deliberate decision.
    assert load_config().primary_variant == "early_warning"


@pytest.fixture(scope="module")
def real_df():
    cfg = load_config()
    if not cfg.raw_data_path.exists():
        pytest.skip("Raw dataset not available")
    return binarize_target(load_data(cfg), cfg)


def test_split_preserves_class_proportion(real_df):
    _, _, y_train, y_test = make_split(real_df, "early_warning")
    overall = real_df[TARGET].mean()
    assert y_train.mean() == pytest.approx(overall, abs=0.02)
    assert y_test.mean() == pytest.approx(overall, abs=0.02)


@pytest.mark.parametrize("variant", list(VARIANT_SEMESTERS))
def test_split_uses_declared_feature_columns(real_df, variant):
    X_train, _, _, _ = make_split(real_df, variant)
    assert list(X_train.columns) == feature_columns(variant)
    assert TARGET not in X_train.columns


def test_variants_share_the_same_students(real_df):
    ew_train, ew_test, _, _ = make_split(real_df, "early_warning")
    full_train, full_test, _, _ = make_split(real_df, "full")
    assert ew_train.index.equals(full_train.index)
    assert ew_test.index.equals(full_test.index)
