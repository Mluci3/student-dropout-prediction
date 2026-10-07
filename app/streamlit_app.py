"""Streamlit app — previsão de evasão estudantil (MLET Fase 3).

Três páginas: Predição (com explicação SHAP local), Desempenho do modelo
e EDA. Carrega o pipeline serializado uma única vez. Os campos do formulário
são derivados da variante do modelo salvo (ver ``src/config.py``): o modelo
principal prevê ao fim do 1º semestre, então não pede dados do 2º semestre.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root importable (so `src` resolves on HF Spaces too).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    GRADE_COLUMNS,
    MACRO_COLUMNS,
    SEMESTER_COUNT_COLUMNS,
    SEMESTER_GRADE_COLUMN,
    VARIANT_DESCRIPTIONS,
    load_config,
    semesters,
)
from src.data import load_data  # noqa: E402
from src.predict import (  # noqa: E402
    get_threshold,
    load_metrics,
    load_pipeline,
    predict_one,
    served_variant,
)

st.set_page_config(page_title="Previsão de Evasão Estudantil", page_icon="🎓", layout="wide")

BINARY_LABELS = {
    "NecessidadesEspeciais": "Necessidades especiais",
    "Devedor": "Devedor (mensalidades)",
    "MensalidadesEmDia": "Mensalidades em dia",
    "Bolsista": "Bolsista",
    "International": "Estudante internacional",
}
SEMESTER_ICONS = {1: "📘", 2: "📗"}


@st.cache_data
def get_reference_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_model():
    return load_pipeline()


def _count_label(col: str) -> str:
    base = col.replace("UnidadesCurriculares", "").replace("Semestre", " - ")
    return base


def prediction_page(df: pd.DataFrame) -> None:
    variant = served_variant()
    st.header("🎓 Previsão de risco de evasão")
    st.caption(
        f"Momento da previsão: **{VARIANT_DESCRIPTIONS[variant]}** — cedo o bastante "
        "para a coordenação intervir. Preencha o perfil do estudante: o modelo "
        "retorna a probabilidade de evasão e os principais fatores que explicam a "
        "previsão (SHAP)."
    )

    record: dict = {}

    with st.expander("👤 Dados pessoais e de ingresso", expanded=True):
        cols = st.columns(2)
        for i, c in enumerate(CATEGORICAL_COLUMNS):
            options = sorted(df[c].dropna().unique().tolist())
            default = df[c].mode().iloc[0]
            record[c] = cols[i % 2].selectbox(
                c, options, index=options.index(default) if default in options else 0
            )
        gq = df["QualificacaoAnteriorGrau"]
        record["QualificacaoAnteriorGrau"] = cols[0].slider(
            "Nota da qualificação anterior", float(gq.min()), float(gq.max()),
            float(gq.median()),
        )
        na = df["NotaAdmissao"]
        record["NotaAdmissao"] = cols[1].slider(
            "Nota de admissão", float(na.min()), float(na.max()), float(na.median())
        )
        bcols = st.columns(len(BINARY_COLUMNS))
        for i, c in enumerate(BINARY_COLUMNS):
            yes = bcols[i].toggle(BINARY_LABELS[c], value=bool(df[c].mode().iloc[0]))
            record[c] = int(yes)

    def academic_block(title, grade_col, count_cols):
        with st.expander(title, expanded=True):
            gcol = df[grade_col].map(
                lambda v: v if v <= 20 else None
            ).dropna()
            record[grade_col] = st.slider(
                f"Nota média ({grade_col[-13:]})", 0.0, 20.0,
                float(round(gcol.median(), 1)), 0.1, key=grade_col,
            )
            ccols = st.columns(len(count_cols))
            for i, c in enumerate(count_cols):
                record[c] = ccols[i].number_input(
                    _count_label(c), min_value=0, max_value=50,
                    value=int(df[c].median()), key=c,
                )

    for s in semesters(variant):
        academic_block(
            f"{SEMESTER_ICONS[s]} Desempenho — {s}º semestre",
            SEMESTER_GRADE_COLUMN[s],
            SEMESTER_COUNT_COLUMNS[s],
        )

    with st.expander("🌍 Contexto socioeconômico"):
        mcols = st.columns(3)
        for i, c in enumerate(MACRO_COLUMNS):
            s = df[c]
            record[c] = mcols[i].number_input(
                c, float(s.min()), float(s.max()), float(s.median())
            )

    if st.button("🔮 Prever risco de evasão", type="primary", width="stretch"):
        get_model()  # ensure loaded / cached
        result = predict_one(record)
        proba = result["dropout_proba"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidade de evasão", f"{proba:.1%}")
        c2.metric("Alerta de evasão", "Sim" if result["dropout_pred"] else "Não")
        risk_color = {"Alto": "🔴", "Médio": "🟡", "Baixo": "🟢"}[result["risk"]]
        c3.metric("Nível de risco", f"{risk_color} {result['risk']}")
        st.progress(min(proba, 1.0))
        st.caption(
            f"Probabilidade **calibrada**: entre alunos com perfil parecido, cerca de "
            f"{proba:.0%} evadem. O alerta é emitido a partir de "
            f"{result['threshold']:.0%} — corte escolhido para detectar ao menos 80% "
            "dos alunos que evadem. Risco **Alto** = evasão mais provável do que não (≥ 50%)."
        )

        st.subheader("Por que esta previsão? (fatores SHAP)")
        try:
            from src.explain import explain_one, load_background

            cfg = load_config()
            factors = explain_one(
                get_model(), pd.DataFrame([record]),
                top_n=cfg.explain["top_n"], background=load_background(),
            )
            fdf = pd.DataFrame(factors)
            fdf["impacto"] = fdf["shap_value"].map(lambda v: round(v, 4))
            st.dataframe(
                fdf[["feature", "direction", "impacto"]].rename(
                    columns={"feature": "Fator", "direction": "Efeito",
                             "impacto": "Impacto (SHAP)"}
                ),
                hide_index=True, width="stretch",
            )
            st.caption(
                "Impacto positivo empurra para evasão; negativo reduz o risco."
            )
        except Exception as exc:  # noqa: BLE001
            st.info(f"Explicação SHAP indisponível: {exc}")


def performance_page() -> None:
    st.header("📊 Desempenho do modelo")
    metrics = load_metrics()
    if not metrics:
        st.warning("Métricas não encontradas. Rode `python -m src.train`.")
        return

    st.subheader(f"Modelo selecionado: `{metrics['selected_model']}`")
    st.caption(f"Momento da previsão: {VARIANT_DESCRIPTIONS[metrics['variant']]}")
    tm = metrics["test_metrics"]
    cols = st.columns(5)
    cols[0].metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    cols[1].metric("PR-AUC", f"{tm['pr_auc']:.3f}")
    cols[2].metric("F1 (evasão)", f"{tm['f1']:.3f}")
    cols[3].metric("Recall (evasão)", f"{tm['recall']:.3f}")
    cols[4].metric("Precisão (evasão)", f"{tm['precision']:.3f}")
    strategy = metrics.get("threshold_strategy", {})
    st.caption(
        f"Threshold de decisão: {get_threshold():.3f} "
        f"(recall-alvo {strategy.get('target_recall', '—')} nas probabilidades "
        "out-of-fold do treino). Probabilidades calibradas."
    )

    of = metrics["overfitting"]
    st.info(f"**Análise over/underfitting:** {of['verdict']}")

    ref = metrics.get("reference")
    if ref:
        st.markdown("#### Por que prever ao fim do 1º semestre?")
        rows = []
        for label, m in (("Principal (em uso)", metrics), ("Referência — teto", ref)):
            t = m["test_metrics"]
            rows.append({
                "Modelo": label,
                "Momento da previsão": VARIANT_DESCRIPTIONS[m["variant"]],
                "ROC-AUC": round(t["roc_auc"], 3),
                "F1": round(t["f1"], 3),
                "Recall": round(t["recall"], 3),
                "Precisão": round(t["precision"], 3),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Esperar o 2º semestre melhora pouco o ROC-AUC, mas nessa altura boa parte "
            "das evasões já aconteceu (ex.: alunos com 0 aprovações no 2º semestre). "
            "O modelo em uso troca essa pequena diferença pela chance de intervir a tempo."
        )

    figs_dir = load_config().figures_dir
    pairs = [
        ("roc_curve.png", "pr_curve.png"),
        ("confusion_matrix.png", "learning_curve.png"),
        ("calibration_curve.png", "overfitting_comparison.png"),
        ("shap_summary.png", "shap_importance_bar.png"),
    ]
    for left, right in pairs:
        c1, c2 = st.columns(2)
        for col, name in ((c1, left), (c2, right)):
            p = figs_dir / name
            if p.exists():
                col.image(str(p), width="stretch")


def eda_page(df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.header("🔎 Análise exploratória (resumo)")
    d = df.copy()
    d["Evasão"] = (d["Target"] == "Desistente").astype(int)
    st.metric("Taxa de evasão global", f"{d['Evasão'].mean():.1%}")

    def rate_plot(col, labels=None):
        rates = d.groupby(col)["Evasão"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.barh([str(x) for x in rates.index], rates.values, color="#4C78A8")
        ax.set_xlabel("Taxa de evasão")
        ax.set_title(f"Evasão por {col}")
        fig.tight_layout()
        return fig

    c1, c2 = st.columns(2)
    c1.pyplot(rate_plot("Devedor"))
    c2.pyplot(rate_plot("Bolsista"))

    from src.features import GradeRepairTransformer

    repaired = GradeRepairTransformer().transform(d[GRADE_COLUMNS])
    d["Nota1"] = repaired[GRADE_COLUMNS[0]]
    fig, ax = plt.subplots(figsize=(6, 3))
    for label, grp in d.groupby("Target"):
        ax.hist(grp["Nota1"], bins=20, alpha=0.5, label=label)
    ax.set_xlabel("Nota média 1º semestre (reparada)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição de notas por desfecho")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


def main() -> None:
    st.sidebar.title("🎓 Evasão Estudantil")
    st.sidebar.caption("MLET — Fase 3 · Alerta precoce ao fim do 1º semestre")
    page = st.sidebar.radio(
        "Navegação", ["Predição", "Desempenho do modelo", "Análise exploratória"]
    )
    df = get_reference_data()
    if page == "Predição":
        prediction_page(df)
    elif page == "Desempenho do modelo":
        performance_page()
    else:
        eda_page(df)


if __name__ == "__main__":
    main()
