"""Feature engineering: grade repair, derived features and the preprocessor.

Every transformation lives inside a scikit-learn ``Pipeline`` so that the exact
same steps run at training time and inside the Streamlit app from a single
serialized artifact. Custom transformers return DataFrames so that column names
flow through to the final ``ColumnTransformer`` (needed for readable SHAP
explanations).

The preprocessor is built per variant (see ``src/config.py``): the early-warning
variant never sees 2nd-semester columns, so it cannot leak future information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    GRADE_COLUMNS,
    SEMESTER_GRADE_COLUMN,
    grade_columns,
    numeric_columns,
    semesters,
)

# Valid Portuguese grading scale upper bound (grades live in [0, 20]).
MAX_VALID_GRADE = 20.0


def derived_columns(semesters_used: tuple[int, ...]) -> list[str]:
    """Names of the engineered columns produced for the given semesters."""
    cols = []
    for s in semesters_used:
        cols += [f"aprovacao_ratio_{s}", f"avaliacao_ratio_{s}"]
    if {1, 2} <= set(semesters_used):
        cols += ["grade_delta", "total_aprovado", "total_inscrito", "aprovacao_ratio_total"]
    return cols


def _repair_grade(value: float) -> float:
    """Recover a single grade whose decimal point was shifted on export.

    ~40% of grade values were stored with the decimal point removed (e.g. a
    grade of 13.43 appears as 1.34e16 or 13875.0). Dividing by 10 until the
    value falls back into the valid [0, 20] range recovers every value, with no
    ambiguous cases in this dataset.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    x = float(value)
    # Guard against pathological input; the real data needs at most ~16 steps.
    for _ in range(40):
        if x <= MAX_VALID_GRADE:
            break
        x /= 10.0
    return x


class GradeRepairTransformer(BaseEstimator, TransformerMixin):
    """Repair the shifted-decimal corruption in the grade columns.

    With ``grade_columns=None`` it repairs whichever known grade columns are
    present, so the same transformer serves every variant.
    """

    def __init__(self, grade_columns: list[str] | None = None):
        self.grade_columns = grade_columns

    def fit(self, X, y=None):  # noqa: D102 - stateless
        return self

    def transform(self, X):
        X = X.copy()
        cols = self.grade_columns
        if cols is None:
            cols = [c for c in GRADE_COLUMNS if c in X.columns]
        for col in cols:
            X[col] = X[col].map(_repair_grade).astype(float)
        return X


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """Element-wise ratio with 0 where the denominator is 0."""
    num = numerator.astype(float).to_numpy()
    den = denominator.astype(float).to_numpy()
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _unit(s: int, field: str) -> str:
    return f"UnidadesCurriculares{s}Semestre{field}"


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Add ratio / delta / total features from the per-semester columns.

    Only the semesters in ``semesters`` are used; cross-semester features
    (delta and totals) exist only when both semesters are available.
    Assumes grades have already been repaired by ``GradeRepairTransformer``.
    """

    def __init__(self, semesters: tuple[int, ...] = (1,)):
        self.semesters = semesters

    def fit(self, X, y=None):  # noqa: D102 - stateless
        return self

    def transform(self, X):
        X = X.copy()
        for s in self.semesters:
            X[f"aprovacao_ratio_{s}"] = _safe_ratio(X[_unit(s, "Aprovado")], X[_unit(s, "Inscrito")])
            X[f"avaliacao_ratio_{s}"] = _safe_ratio(X[_unit(s, "Avaliacoes")], X[_unit(s, "Inscrito")])
        if {1, 2} <= set(self.semesters):
            g1, g2 = SEMESTER_GRADE_COLUMN[1], SEMESTER_GRADE_COLUMN[2]
            X["grade_delta"] = X[g2].astype(float) - X[g1].astype(float)
            X["total_aprovado"] = X[_unit(1, "Aprovado")].astype(float) + X[_unit(2, "Aprovado")].astype(float)
            X["total_inscrito"] = X[_unit(1, "Inscrito")].astype(float) + X[_unit(2, "Inscrito")].astype(float)
            X["aprovacao_ratio_total"] = _safe_ratio(X["total_aprovado"], X["total_inscrito"])
        return X


def build_preprocessor(variant: str) -> Pipeline:
    """Build the preprocessing pipeline for a variant (shared by models and app).

    Steps: grade repair -> derived features -> column transform (one-hot
    categoricals, scale numerics, pass binaries). Sensitive personal attributes
    are not part of any variant (see ``SENSITIVE_COLUMNS`` in ``src/config.py``).
    """
    sems = semesters(variant)
    numeric_features = (
        numeric_columns(variant) + grade_columns(variant) + derived_columns(sems)
    )

    column_transform = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
            ("num", StandardScaler(), numeric_features),
            ("bin", "passthrough", BINARY_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("grade_repair", GradeRepairTransformer()),
            ("derived", DerivedFeatures(semesters=sems)),
            ("column_transform", column_transform),
        ]
    )
