"""Train, cross-validate, tune, calibrate, evaluate and serialize the model.

Run with:  ``python -m src.train``

Two variants are trained with the exact same protocol and the same
train/test students (see ``src/config.py``):

- the PRIMARY variant (early warning, end of 1st semester) is serialized and
  served by the app, with all diagnostic figures;
- the REFERENCE variant (both semesters) is reported in ``metrics.json`` only,
  as a performance ceiling.

Protocol per variant:
  1. Tune each candidate (LogReg, RandomForest, HistGradientBoosting) with
     RandomizedSearchCV under identical preprocessing, then select with the
     one-standard-error rule: among the candidates whose tuned CV ROC-AUC is
     within one standard error of the best, keep the simplest (train vs CV scores
     of every candidate feed the overfitting table).
  2. Calibrate the selected pipeline's probabilities (CalibratedClassifierCV).
  3. Choose the decision threshold for a target recall on out-of-fold
     calibrated probabilities of the training set (the test set is never used).
  4. Evaluate on the held-out test set: metrics + bootstrap CIs, calibration,
     baselines and a fairness audit by group.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .config import (
    FAIRNESS_GROUP_COLUMN,
    VARIANT_DESCRIPTIONS,
    feature_columns,
    load_config,
    semesters,
)
from .data import binarize_target, load_data, make_split
from .evaluate import (
    bootstrap_ci,
    choose_threshold_for_recall,
    compute_metrics,
    group_metrics,
    plot_calibration,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_overfitting_comparison,
    plot_roc_pr,
)
from .explain import save_global_summary
from .features import build_preprocessor


def _candidates(variant: str, seed: int) -> dict[str, tuple[Pipeline, dict]]:
    """Return {name: (pipeline, param_distributions)} for each candidate model."""
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("preprocess", build_preprocessor(variant)),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced", max_iter=2000, random_state=seed
                        ),
                    ),
                ]
            ),
            {"model__C": loguniform(1e-2, 1e2)},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("preprocess", build_preprocessor(variant)),
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced", random_state=seed, n_jobs=-1
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": randint(200, 600),
                "model__max_depth": randint(4, 20),
                "model__min_samples_leaf": randint(1, 20),
                "model__max_features": ["sqrt", "log2", None],
            },
        ),
        "hist_gradient_boosting": (
            Pipeline(
                [
                    ("preprocess", build_preprocessor(variant)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            class_weight="balanced", random_state=seed
                        ),
                    ),
                ]
            ),
            {
                "model__learning_rate": loguniform(1e-2, 3e-1),
                "model__max_leaf_nodes": randint(15, 63),
                "model__max_depth": randint(3, 12),
                "model__l2_regularization": loguniform(1e-3, 1e1),
                "model__min_samples_leaf": randint(10, 60),
            },
        ),
    }


# Candidates from simplest to most complex (used to break statistical ties).
COMPLEXITY_ORDER = ["logistic_regression", "hist_gradient_boosting", "random_forest"]


def _select_one_se(comparison: dict[str, dict], n_folds: int) -> dict:
    """One-standard-error rule: simplest model statistically tied with the best.

    Picking the raw maximum among tied scores tends to pick the most complex
    (and most overfitted) model for no real gain.
    """
    best = max(comparison, key=lambda n: comparison[n]["cv_auc"])
    tolerance = comparison[best]["cv_std"] / np.sqrt(n_folds)
    floor = comparison[best]["cv_auc"] - tolerance
    eligible = [n for n in COMPLEXITY_ORDER if comparison[n]["cv_auc"] >= floor]
    return {
        "rule": "one_standard_error",
        "highest_cv_model": best,
        "tolerance": float(tolerance),
        "eligible": eligible,
        "selected": eligible[0],
    }


def _rule_baseline(X, variant: str) -> tuple[str, np.ndarray]:
    """Simple rule a coordinator could apply by hand, using only the variant's data."""
    last = max(semesters(variant))
    description = f"mensalidades atrasadas OU 0 disciplinas aprovadas no {last}º semestre"
    flag = (X["MensalidadesEmDia"] == 0) | (X[f"UnidadesCurriculares{last}SemestreAprovado"] == 0)
    return description, flag.astype(int).to_numpy()


def _overfitting_verdict(train_auc: float, cv_auc: float, test_auc: float) -> str:
    # Rules of thumb: a train-CV gap above 0.05 ROC-AUC is larger than the
    # fold-to-fold spread (~0.01) by a wide margin; a CV ROC-AUC below 0.7 means
    # the model barely separates the classes.
    gap = train_auc - cv_auc
    if cv_auc < 0.7:
        return (
            "Possível underfitting: desempenho de validação baixo "
            f"(CV ROC-AUC = {cv_auc:.3f})."
        )
    if gap > 0.05:
        return (
            f"Sinais de overfitting: gap treino-CV = {gap:.3f} "
            f"(treino {train_auc:.3f} vs CV {cv_auc:.3f})."
        )
    return (
        "Sem overfitting/underfitting relevante: treino, CV e teste próximos "
        f"(treino {train_auc:.3f}, CV {cv_auc:.3f}, teste {test_auc:.3f})."
    )


def _jsonable(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def fit_variant(variant, X_train, X_test, y_train, y_test, groups_test, cfg, cv) -> dict:
    """Run the full tuning -> selection -> calibration -> evaluation protocol."""
    seed = cfg.seed
    tag = f"[{variant}]"

    # --- Tune every candidate, then select by tuned CV ROC-AUC --------------
    comparison: dict[str, dict] = {}
    searches = {}
    for name, (pipe, dist) in _candidates(variant, seed).items():
        search = RandomizedSearchCV(
            pipe,
            dist,
            n_iter=cfg.train["n_iter_search"],
            scoring=cfg.train["scoring"],
            cv=cv,
            random_state=seed,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        i = search.best_index_
        res = search.cv_results_
        comparison[name] = {
            "cv_auc": float(res["mean_test_score"][i]),
            "cv_std": float(res["std_test_score"][i]),
            "train_auc": float(res["mean_train_score"][i]),
            "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        }
        searches[name] = search
        c = comparison[name]
        print(f"{tag} {name:<24} CV ROC-AUC = {c['cv_auc']:.4f} ± {c['cv_std']:.4f} "
              f"(treino {c['train_auc']:.3f})")
    selection = _select_one_se(comparison, cfg.train["cv_folds"])
    best_name = selection["selected"]
    base = searches[best_name].best_estimator_
    print(f"{tag} Maior CV: {selection['highest_cv_model']} | empatados (1-SE, tol "
          f"{selection['tolerance']:.4f}): {selection['eligible']} -> selecionado: {best_name}")

    # --- Probability calibration --------------------------------------------
    calibrated_template = CalibratedClassifierCV(
        clone(base), method=cfg.train["calibration_method"], cv=cv, ensemble=False
    )
    calibrated = clone(calibrated_template).fit(X_train, y_train)

    # --- Threshold for the target recall on out-of-fold probabilities --------
    oof = cross_val_predict(
        calibrated_template, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    target = cfg.train["target_recall"]
    threshold = choose_threshold_for_recall(y_train, oof, target)

    # --- Held-out test evaluation -------------------------------------------
    proba = calibrated.predict_proba(X_test)[:, 1]
    proba_raw = base.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, proba, threshold)
    ci = bootstrap_ci(y_test, proba, threshold, cfg.train["bootstrap_samples"], seed)

    rule_desc, rule_pred = _rule_baseline(X_test, variant)
    dummy = DummyClassifier(strategy="stratified", random_state=seed).fit(X_train, y_train)
    baselines = {
        "rule": {"description": rule_desc, **compute_metrics(y_test, rule_pred, 0.5)},
        "dummy_stratified": compute_metrics(y_test, dummy.predict_proba(X_test)[:, 1], 0.5),
    }

    sel = comparison[best_name]
    verdict = _overfitting_verdict(sel["train_auc"], sel["cv_auc"], test_metrics["roc_auc"])
    print(f"{tag} Threshold = {threshold:.3f} (recall-alvo {target}) | Teste ROC-AUC = "
          f"{test_metrics['roc_auc']:.4f} | Recall = {test_metrics['recall']:.4f} | "
          f"Precisão = {test_metrics['precision']:.4f}")
    print(f"{tag} Baseline regra: F1 = {baselines['rule']['f1']:.4f} | {verdict}")

    return {
        "calibrated": calibrated,
        "base": base,
        "proba": proba,
        "proba_raw": proba_raw,
        "summary": {
            "variant": variant,
            "prediction_moment": VARIANT_DESCRIPTIONS[variant],
            "selected_model": best_name,
            "best_params": sel["best_params"],
            "model_comparison": comparison,
            "model_selection": selection,
            "threshold": threshold,
            "threshold_strategy": {"criterion": "target_recall", "target_recall": target},
            "calibration": {
                "method": cfg.train["calibration_method"],
                "real_dropout_rate_test": float(np.mean(y_test)),
                "mean_proba_before": float(proba_raw.mean()),
                "mean_proba_after": float(proba.mean()),
                "brier_before": float(brier_score_loss(y_test, proba_raw)),
                "brier_after": float(brier_score_loss(y_test, proba)),
            },
            "test_metrics": test_metrics,
            "test_ci95": ci,
            "baselines": baselines,
            "fairness": {"group_column": FAIRNESS_GROUP_COLUMN,
                         "by_group": group_metrics(y_test, proba, threshold, groups_test)},
            "overfitting": {
                "train_auc": sel["train_auc"],
                "cv_auc": sel["cv_auc"],
                "test_auc": test_metrics["roc_auc"],
                "gap_train_cv": sel["train_auc"] - sel["cv_auc"],
                "verdict": verdict,
            },
        },
    }


def main() -> None:
    cfg = load_config()
    seed = cfg.seed
    cv = StratifiedKFold(n_splits=cfg.train["cv_folds"], shuffle=True, random_state=seed)
    primary, reference = cfg.primary_variant, cfg.reference_variant

    # --- Data (same students in train/test for every variant) --------------
    df = binarize_target(load_data(cfg), cfg)
    splits = {v: make_split(df, v, cfg) for v in (primary, reference)}
    assert splits[primary][0].index.equals(splits[reference][0].index), \
        "As variantes precisam usar exatamente os mesmos alunos de treino/teste."
    X_train, X_test, y_train, y_test = splits[primary]
    groups_test = df.loc[X_test.index, FAIRNESS_GROUP_COLUMN].to_numpy()
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)} | "
          f"Taxa de evasão (treino): {y_train.mean():.3f}")

    results = {
        v: fit_variant(v, *splits[v], groups_test, cfg, cv) for v in (primary, reference)
    }
    main_result = results[primary]
    summary = main_result["summary"]
    proba = main_result["proba"]
    threshold = summary["threshold"]

    # --- Diagnostic figures for the served (primary) model ------------------
    figures_dir = cfg.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    lc = plot_learning_curve(
        clone(main_result["base"]), X_train, y_train, figures_dir / "learning_curve.png", cv, seed
    )
    plot_confusion_matrix(y_test, proba, threshold, figures_dir / "confusion_matrix.png")
    plot_roc_pr(y_test, proba, figures_dir)
    plot_calibration(y_test, main_result["proba_raw"], proba, figures_dir / "calibration_curve.png")
    plot_overfitting_comparison(
        summary["model_comparison"], figures_dir / "overfitting_comparison.png"
    )
    bg = X_train.sample(n=min(cfg.explain["background_size"], len(X_train)), random_state=seed)
    try:
        save_global_summary(main_result["calibrated"], bg, figures_dir)
        print("Figuras SHAP salvas.")
    except Exception as exc:  # noqa: BLE001 - SHAP is best-effort for the report
        print(f"Aviso: não foi possível gerar SHAP global ({exc}).")

    # --- Persist artifacts --------------------------------------------------
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(main_result["calibrated"], cfg.model_path)
    joblib.dump(bg, cfg.background_path)  # SHAP reference background for the app

    summary["overfitting"]["learning_curve"] = lc
    metrics = {
        **summary,
        "feature_columns": feature_columns(primary),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dropout_rate": float(df["dropout"].mean()),
        "reference": results[reference]["summary"],
    }
    with open(cfg.metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    ref = results[reference]["summary"]["test_metrics"]
    print(f"\nPrincipal ({primary}): teste ROC-AUC = {summary['test_metrics']['roc_auc']:.4f} | "
          f"Referência ({reference}): {ref['roc_auc']:.4f}")
    print(f"Modelo salvo em {cfg.model_path}")
    print(f"Métricas salvas em {cfg.metrics_path}")


if __name__ == "__main__":
    main()
