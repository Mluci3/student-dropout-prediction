"""Offline monitoring of a new batch of students against the training data.

Run with:  ``python -m src.monitor path/to/new_students.xlsx [--output report.csv]``

The batch must have the same raw columns as ``data/raw/StudentsPrepared.xlsx``
(the ``Target`` column is optional). Three checks are reported:

1. **Data quality** — missing values, corrupted grades (> 20, the shifted-decimal
   problem found in the training data) and categories never seen in training.
2. **Data drift** — Population Stability Index (PSI) of every model input,
   comparing the batch with the training split. Rule of thumb used in industry:
   PSI < 0.10 stable, 0.10–0.25 attention, > 0.25 significant shift.
3. **Prediction drift** — alert rate and mean predicted probability of the served
   model on the batch, next to the values observed on the test set.

Ground truth (who actually dropped out) only arrives months later, so these checks
are the early signals that the model may need retraining. See ``docs/monitoring.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BINARY_COLUMNS, CATEGORICAL_COLUMNS, GRADE_COLUMNS, load_config
from .data import binarize_target, load_data, make_split
from .features import MAX_VALID_GRADE, GradeRepairTransformer
from .predict import get_threshold, load_metrics, load_pipeline, model_features, risk_label

PSI_ATTENTION = 0.10
PSI_ALERT = 0.25
_EPS = 1e-4  # avoids log(0) for empty bins / unseen categories


def _psi(ref_share: np.ndarray, new_share: np.ndarray) -> float:
    ref_share = np.clip(ref_share, _EPS, None)
    new_share = np.clip(new_share, _EPS, None)
    return float(np.sum((new_share - ref_share) * np.log(new_share / ref_share)))


def psi_numeric(reference: pd.Series, new: pd.Series, bins: int = 10) -> float:
    """PSI with quantile bins taken from the reference distribution."""
    ref = reference.dropna().astype(float).to_numpy()
    cur = new.dropna().astype(float).to_numpy()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    if len(edges) < 3:  # (almost) constant feature: compare as categories
        return psi_categorical(reference.astype(str), new.astype(str))
    ref_share = np.histogram(ref, edges)[0] / len(ref)
    new_share = np.histogram(cur, edges)[0] / max(len(cur), 1)
    return _psi(ref_share, new_share)


def psi_categorical(reference: pd.Series, new: pd.Series) -> float:
    """PSI over the union of categories (unseen categories get a tiny reference share)."""
    ref = reference.value_counts(normalize=True)
    cur = new.value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    return _psi(ref.reindex(cats, fill_value=0).to_numpy(), cur.reindex(cats, fill_value=0).to_numpy())


def _status(psi: float) -> str:
    if psi > PSI_ALERT:
        return "alerta"
    if psi > PSI_ATTENTION:
        return "atenção"
    return "ok"


def data_quality(reference: pd.DataFrame, new: pd.DataFrame, features: list[str]) -> dict:
    """Missing columns/values, corrupted grades and unseen categories in the batch."""
    missing_columns = [c for c in features if c not in new.columns]
    present = [c for c in features if c in new.columns]
    report = {
        "n_rows": int(len(new)),
        "missing_columns": missing_columns,
        "missing_values": {c: int(n) for c, n in new[present].isna().sum().items() if n},
        "corrupted_grade_share": {
            c: float((new[c] > MAX_VALID_GRADE).mean()) for c in GRADE_COLUMNS if c in present
        },
        "unseen_categories": {},
    }
    for c in CATEGORICAL_COLUMNS:
        if c in present:
            unseen = sorted(set(new[c].dropna()) - set(reference[c].dropna()))
            if unseen:
                report["unseen_categories"][c] = unseen
    return report


def drift_report(reference: pd.DataFrame, new: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """PSI and status for every model input (grades are compared after repair)."""
    repair = GradeRepairTransformer()
    ref, cur = repair.transform(reference[features]), repair.transform(new[features])
    rows = []
    for c in features:
        categorical = c in CATEGORICAL_COLUMNS or c in BINARY_COLUMNS
        psi = psi_categorical(ref[c], cur[c]) if categorical else psi_numeric(ref[c], cur[c])
        rows.append({
            "feature": c,
            "type": "categórica" if categorical else "numérica",
            "psi": round(psi, 4),
            "status": _status(psi),
        })
    order = {"alerta": 0, "atenção": 1, "ok": 2}
    return (pd.DataFrame(rows)
            .sort_values(["status", "psi"], key=lambda s: s.map(order) if s.name == "status" else -s)
            .reset_index(drop=True))


def prediction_summary(new: pd.DataFrame) -> dict:
    """How the served model behaves on the batch, next to the test-set reference."""
    proba = load_pipeline().predict_proba(new[model_features()])[:, 1]
    threshold = get_threshold()
    risks = pd.Series([risk_label(p, threshold) for p in proba]).value_counts(normalize=True)
    test = load_metrics().get("test_metrics", {})
    return {
        "n_rows": int(len(proba)),
        "mean_proba": float(proba.mean()),
        "alert_rate": float((proba >= threshold).mean()),
        "risk_share": {k: float(risks.get(k, 0.0)) for k in ("Baixo", "Médio", "Alto")},
        "reference_test_mean_proba": test.get("mean_predicted_proba"),
        "reference_test_alert_rate": test.get("alert_rate"),
    }


def load_reference() -> pd.DataFrame:
    """Training split of the served variant (raw values, as the model received them)."""
    cfg = load_config()
    X_train, _, _, _ = make_split(binarize_target(load_data(cfg), cfg), cfg.primary_variant, cfg)
    return X_train


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relatório de qualidade e drift de um lote novo.")
    parser.add_argument("batch", type=Path, help="planilha (.xlsx) ou .csv com os alunos novos")
    parser.add_argument("--output", type=Path, help="CSV para salvar a tabela de drift")
    args = parser.parse_args()

    features = model_features()
    reference, new = load_reference(), _read(args.batch)

    quality = data_quality(reference, new, features)
    print(f"Lote: {quality['n_rows']} alunos")
    if quality["missing_columns"]:
        print(f"ERRO: colunas ausentes no lote: {quality['missing_columns']}")
        raise SystemExit(1)
    print("\n== Qualidade dos dados ==")
    print(f"Valores ausentes: {quality['missing_values'] or 'nenhum'}")
    print(f"Notas corrompidas (> 20, antes do reparo): "
          f"{ {k: f'{v:.1%}' for k, v in quality['corrupted_grade_share'].items()} }")
    print(f"Categorias nunca vistas no treino: {quality['unseen_categories'] or 'nenhuma'}")

    report = drift_report(reference, new, features)
    print("\n== Drift por variável (PSI) ==")
    print(report.to_string(index=False))
    counts = report["status"].value_counts().to_dict()
    print(f"Resumo: {counts}")

    summary = prediction_summary(new)
    print("\n== Previsões do modelo no lote ==")
    print(f"Probabilidade média: {summary['mean_proba']:.3f} "
          f"(teste: {summary['reference_test_mean_proba']:.3f})")
    print(f"Taxa de alerta: {summary['alert_rate']:.1%} "
          f"(teste: {summary['reference_test_alert_rate']:.1%})")
    print(f"Distribuição de risco: { {k: f'{v:.1%}' for k, v in summary['risk_share'].items()} }")

    if args.output:
        report.to_csv(args.output, index=False)
        print(f"\nTabela de drift salva em {args.output}")


if __name__ == "__main__":
    main()
