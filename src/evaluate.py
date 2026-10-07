"""Metrics, threshold selection, confidence intervals and diagnostic plots.

Plots use the non-interactive ``Agg`` backend so this module is safe to import
from scripts and from Streamlit alike.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve  # noqa: E402


def choose_threshold_for_recall(y_true, proba, target_recall: float) -> float:
    """Highest threshold whose recall on the positive class is >= ``target_recall``.

    In an early-warning system a missed dropout (false negative) costs more than
    an extra alert, so the cutoff is set by the share of dropouts we want to
    catch rather than by F1. Because the criterion only depends on the ranking of
    the probabilities, it is unaffected by (monotonic) calibration.
    """
    _, recalls, thresholds = precision_recall_curve(y_true, proba)
    # recalls[i] is the recall when predicting positive for proba >= thresholds[i];
    # recall decreases as the index (threshold) increases.
    ok = np.where(recalls[:-1] >= target_recall)[0]
    return float(thresholds[ok.max()]) if len(ok) else float(thresholds.min())


def compute_metrics(y_true, proba, threshold: float) -> dict:
    """Return the core binary-classification metrics as a plain dict."""
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "brier": float(brier_score_loss(y_true, proba)),
        "mean_predicted_proba": float(proba.mean()),
        "alert_rate": float(y_pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci(y_true, proba, threshold: float, n_samples: int, seed: int) -> dict:
    """95% bootstrap confidence intervals for the main test metrics."""
    y = np.asarray(y_true)
    proba = np.asarray(proba)
    rng = np.random.default_rng(seed)
    stats = {"roc_auc": [], "f1": [], "recall": [], "precision": []}
    for _ in range(n_samples):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], proba[idx]
        if yb.min() == yb.max():  # degenerate resample with a single class
            continue
        pred = (pb >= threshold).astype(int)
        stats["roc_auc"].append(roc_auc_score(yb, pb))
        stats["f1"].append(f1_score(yb, pred, zero_division=0))
        stats["recall"].append(recall_score(yb, pred, zero_division=0))
        stats["precision"].append(precision_score(yb, pred, zero_division=0))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}


def group_metrics(y_true, proba, threshold: float, groups) -> dict:
    """Recall / precision / alert rate per group (fairness audit)."""
    y = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    groups = np.asarray(groups)
    out = {}
    for g in sorted(set(groups)):
        m = groups == g
        out[str(g)] = {
            "n": int(m.sum()),
            "dropout_rate": float(y[m].mean()),
            "recall": float(recall_score(y[m], pred[m], zero_division=0)),
            "precision": float(precision_score(y[m], pred[m], zero_division=0)),
            "alert_rate": float(pred[m].mean()),
        }
    return out


def plot_confusion_matrix(y_true, proba, threshold: float, out_path: Path) -> None:
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não-evasão", "Evasão"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusão (teste)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc_pr(y_true, proba, out_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title("Curva ROC (teste)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=120)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"PR (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisão")
    ax.set_title("Curva Precisão-Recall (teste)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=120)
    plt.close(fig)


def plot_calibration(y_true, proba_raw, proba_cal, out_path: Path) -> None:
    """Reliability diagram: model before vs after probability calibration."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Calibração perfeita")
    for label, p in (("Antes da calibração", proba_raw), ("Calibrado (em uso)", proba_cal)):
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_true, p)
        ax.plot(mean_pred, frac_pos, "o-", label=f"{label} — Brier {brier:.3f}")
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Fração real de evasão")
    ax.set_title("Curva de calibração (teste)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_overfitting_comparison(rows: dict, out_path: Path) -> None:
    """Train vs cross-validation ROC-AUC for each tuned candidate model."""
    names = list(rows)
    train = [rows[n]["train_auc"] for n in names]
    cv = [rows[n]["cv_auc"] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.18, train, 0.36, label="Treino")
    ax.bar(x + 0.18, cv, 0.36, label="Validação cruzada")
    for i in range(len(names)):
        ax.annotate(f"gap {train[i] - cv[i]:.3f}", (x[i], max(train[i], cv[i]) + 0.005),
                    ha="center", fontsize=8)
    ax.set_xticks(x, [n.replace("_", "\n") for n in names])
    ax.set_ylim(min(cv) - 0.05, 1.03)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Overfitting por modelo (após tuning)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_learning_curve(estimator, X, y, out_path: Path, cv, seed: int) -> dict:
    """Plot and return the learning curve (train vs CV ROC-AUC by train size)."""
    sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 5),
        random_state=seed,
        n_jobs=-1,
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sizes, train_mean, "o-", label="Treino")
    ax.plot(sizes, val_mean, "o-", label="Validação cruzada")
    ax.fill_between(sizes, train_scores.min(axis=1), train_scores.max(axis=1), alpha=0.1)
    ax.fill_between(sizes, val_scores.min(axis=1), val_scores.max(axis=1), alpha=0.1)
    ax.set_xlabel("Tamanho do conjunto de treino")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Curva de aprendizado")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        "train_sizes": [int(s) for s in sizes],
        "train_auc": [float(v) for v in train_mean],
        "cv_auc": [float(v) for v in val_mean],
    }
