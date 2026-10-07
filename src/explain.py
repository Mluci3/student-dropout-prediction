"""Model explainability with SHAP (global summary + per-student local factors).

SHAP values are computed on the *preprocessed* feature space so that labels such
as ``Curso_Enfermagem`` or ``grade_delta`` are human-readable. The functions
here are reused by the training script (to save the global summary figure) and
by the Streamlit app (to explain a single prediction).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from .config import load_config  # noqa: E402

# Friendly Portuguese labels for the engineered / cryptic feature names.
FRIENDLY_LABELS = {
    "aprovacao_ratio_1": "Taxa de aprovação (1º sem.)",
    "aprovacao_ratio_2": "Taxa de aprovação (2º sem.)",
    "avaliacao_ratio_1": "Taxa de avaliações (1º sem.)",
    "avaliacao_ratio_2": "Taxa de avaliações (2º sem.)",
    "grade_delta": "Variação de nota entre semestres",
    "total_aprovado": "Total de disciplinas aprovadas",
    "total_inscrito": "Total de disciplinas inscritas",
    "aprovacao_ratio_total": "Taxa de aprovação (geral)",
}


def base_pipeline(model) -> Pipeline:
    """Return the fitted preprocess+model pipeline behind the served model.

    The served artifact is a ``CalibratedClassifierCV`` (``ensemble=False``) that
    wraps a single pipeline fitted on the whole training set. Calibration is a
    monotonic map of that pipeline's output, so explaining the base pipeline
    gives the same ranking of factors.
    """
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def _split_pipeline(pipeline):
    """Return (preprocess, model) from a fitted (possibly calibrated) pipeline."""
    pipe = base_pipeline(pipeline)
    return pipe.named_steps["preprocess"], pipe.named_steps["model"]


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocess, _ = _split_pipeline(pipeline)
    ct = preprocess.named_steps["column_transform"]
    return list(ct.get_feature_names_out())


def prettify(name: str) -> str:
    """Make a transformed feature name friendlier for display."""
    return FRIENDLY_LABELS.get(name, name)


def _transform(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    preprocess, _ = _split_pipeline(pipeline)
    return np.asarray(preprocess.transform(X))


def load_background() -> pd.DataFrame | None:
    """Load the saved raw background sample used as the SHAP reference."""
    cfg = load_config()
    if cfg.background_path.exists():
        return joblib.load(cfg.background_path)
    return None


def build_explainer(pipeline: Pipeline, X_background: pd.DataFrame):
    """Create a SHAP explainer over the preprocessed feature space.

    Picks the fast exact explainer for the model family:
    ``LinearExplainer`` for linear models, ``TreeExplainer`` for tree/boosting
    models, and the model-agnostic explainer (sampled background) otherwise.
    """
    _, model = _split_pipeline(pipeline)
    bg = _transform(pipeline, X_background)
    if hasattr(model, "coef_"):  # linear models (e.g. LogisticRegression)
        return shap.LinearExplainer(model, bg)
    try:
        return shap.TreeExplainer(model)
    except Exception:  # noqa: BLE001 - fall back to the generic explainer
        return shap.Explainer(model.predict_proba, bg)


def _positive_class_shap(explainer, Z: np.ndarray) -> np.ndarray:
    """Return a (n_samples, n_features) SHAP matrix for the positive class."""
    values = explainer.shap_values(Z)
    # Tree models may return a list [class0, class1] or a 3D array.
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def save_global_summary(
    pipeline: Pipeline, X_background: pd.DataFrame, out_dir: Path, max_display: int = 15
) -> None:
    """Save the global SHAP summary (beeswarm) and mean-|SHAP| bar figures."""
    feature_names = [prettify(n) for n in get_feature_names(pipeline)]
    Z = _transform(pipeline, X_background)
    explainer = build_explainer(pipeline, X_background)
    shap_matrix = _positive_class_shap(explainer, Z)

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_matrix, Z, feature_names=feature_names, max_display=max_display, show=False
    )
    plt.title("Importância das features (SHAP) — classe evasão")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_matrix,
        Z,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP médio absoluto por feature")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance_bar.png", dpi=120, bbox_inches="tight")
    plt.close()


def explain_one(
    pipeline: Pipeline,
    x_row: pd.DataFrame,
    top_n: int = 6,
    background: pd.DataFrame | None = None,
) -> list[dict]:
    """Return the top-N features pushing a single prediction, by |SHAP value|.

    Each item: ``{"feature", "shap_value", "direction"}`` where a positive value
    increases the predicted dropout probability. A representative ``background``
    is required for exact SHAP values; if none is given it is loaded from disk,
    falling back to the row itself only as a last resort.
    """
    feature_names = get_feature_names(pipeline)
    Z = _transform(pipeline, x_row)
    bg = background if background is not None else load_background()
    explainer = build_explainer(pipeline, bg if bg is not None else x_row)
    shap_row = _positive_class_shap(explainer, Z)[0]

    order = np.argsort(np.abs(shap_row))[::-1][:top_n]
    factors = []
    for idx in order:
        value = float(shap_row[idx])
        factors.append(
            {
                "feature": prettify(feature_names[idx]),
                "shap_value": value,
                "direction": "aumenta o risco" if value > 0 else "reduz o risco",
            }
        )
    return factors
