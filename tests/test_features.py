"""Tests for grade repair, derived features and the per-variant preprocessor."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.config import load_config
from src.data import binarize_target, load_data, split_xy
from src.features import (
    DerivedFeatures,
    GradeRepairTransformer,
    _repair_grade,
    build_preprocessor,
    derived_columns,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (1.34285714285714e16, 13.4285714285714),  # scientific-notation corruption
        (13875.0, 13.875),                         # x1000 corruption
        (14545.0, 14.545),
        (13.4, 13.4),                              # already valid -> untouched
        (0.0, 0.0),
        (18.88, 18.88),
        (20.0, 20.0),                              # boundary stays
    ],
)
def test_repair_grade_values(raw, expected):
    assert _repair_grade(raw) == pytest.approx(expected, rel=1e-9)


def test_repair_grade_preserves_nan():
    assert np.isnan(_repair_grade(float("nan")))


def test_grade_repair_transformer_within_range():
    df = pd.DataFrame(
        {
            "UnidadesCurriculares1SemestreGrau": [1.34e16, 13875.0, 12.5],
            "UnidadesCurriculares2SemestreGrau": [1.85e16, 0.0, 14.2],
        }
    )
    out = GradeRepairTransformer().transform(df)
    assert (out <= 20).all().all()
    assert (out >= 0).all().all()


def test_grade_repair_handles_only_first_semester():
    # The early-warning variant has no 2nd-semester grade column.
    df = pd.DataFrame({"UnidadesCurriculares1SemestreGrau": [13875.0, 12.5]})
    out = GradeRepairTransformer().transform(df)
    assert list(out.columns) == ["UnidadesCurriculares1SemestreGrau"]
    assert out.iloc[0, 0] == pytest.approx(13.875)


def test_grade_repair_follows_sklearn_param_convention():
    # __init__ must store parameters untouched so clone/get_params round-trip.
    assert clone(GradeRepairTransformer()).get_params()["grade_columns"] is None


def test_derived_features_safe_ratio_and_delta():
    df = pd.DataFrame(
        {
            "UnidadesCurriculares1SemestreGrau": [12.0],
            "UnidadesCurriculares2SemestreGrau": [14.0],
            "UnidadesCurriculares1SemestreInscrito": [0],  # triggers divide-by-zero
            "UnidadesCurriculares2SemestreInscrito": [6],
            "UnidadesCurriculares1SemestreAprovado": [0],
            "UnidadesCurriculares2SemestreAprovado": [3],
            "UnidadesCurriculares1SemestreAvaliacoes": [0],
            "UnidadesCurriculares2SemestreAvaliacoes": [5],
        }
    )
    out = DerivedFeatures(semesters=(1, 2)).transform(df)
    assert out["aprovacao_ratio_1"].iloc[0] == 0.0          # 0/0 -> 0, no crash
    assert out["aprovacao_ratio_2"].iloc[0] == pytest.approx(0.5)
    assert out["grade_delta"].iloc[0] == pytest.approx(2.0)
    assert out["total_inscrito"].iloc[0] == 6
    assert out["aprovacao_ratio_total"].iloc[0] == pytest.approx(0.5)


def test_derived_features_first_semester_only():
    df = pd.DataFrame(
        {
            "UnidadesCurriculares1SemestreInscrito": [6],
            "UnidadesCurriculares1SemestreAprovado": [3],
            "UnidadesCurriculares1SemestreAvaliacoes": [6],
        }
    )
    out = DerivedFeatures(semesters=(1,)).transform(df)
    added = [c for c in out.columns if c not in df.columns]
    assert added == derived_columns((1,)) == ["aprovacao_ratio_1", "avaliacao_ratio_1"]
    assert out["aprovacao_ratio_1"].iloc[0] == pytest.approx(0.5)


@pytest.fixture(scope="module")
def real_df():
    cfg = load_config()
    if not cfg.raw_data_path.exists():
        pytest.skip("Raw dataset not available")
    return binarize_target(load_data(cfg), cfg)


def test_early_warning_model_matrix_has_no_second_semester_information(real_df):
    """Leakage guard: nothing from the 2nd semester may reach the primary model."""
    X, _ = split_xy(real_df, "early_warning")
    pre = build_preprocessor("early_warning").fit(X)
    names = list(pre.named_steps["column_transform"].get_feature_names_out())
    second_semester = set(derived_columns((1, 2))) - set(derived_columns((1,)))
    assert not [n for n in names if "2Semestre" in n or n in second_semester]
    assert pre.transform(X).shape == (len(X), len(names))
