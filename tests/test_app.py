"""End-to-end smoke tests of the Streamlit app (headless, via streamlit.testing).

They run the real app script against the committed model artifact, click the
predict button and visit every page, failing on any exception shown to users.
"""

import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT, SENSITIVE_COLUMNS, load_config

APP = str(ROOT / "app" / "streamlit_app.py")
PAGES = ["Predição", "Desempenho do modelo", "Análise exploratória"]


@pytest.fixture
def app():
    cfg = load_config()
    if not (cfg.model_path.exists() and cfg.raw_data_path.exists()):
        pytest.skip("Model artifact or dataset not available")
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    return at


def _labels(at):
    widgets = [*at.selectbox, *at.slider, *at.number_input, *at.toggle]
    return [w.label for w in widgets]


def test_form_only_asks_for_data_available_at_prediction_time(app):
    labels = _labels(app)
    assert labels, "the prediction form should have input fields"
    # Early-warning model: nothing from the 2nd semester, no sensitive attributes.
    assert not [lbl for lbl in labels if "2Semestre" in lbl or "2 - " in lbl]
    assert not set(SENSITIVE_COLUMNS) & set(labels)


def test_predict_shows_probability_risk_and_explanation(app):
    app.button[0].click().run()
    assert not app.exception, app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Probabilidade de evasão"].endswith("%")
    assert metrics["Nível de risco"].split()[-1] in {"Baixo", "Médio", "Alto"}
    assert metrics["Alerta de evasão"] in {"Sim", "Não"}
    assert len(app.dataframe[0].value) == load_config().explain["top_n"]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, (page, app.exception)


def test_performance_page_shows_all_figures(app):
    app.sidebar.radio[0].set_value("Desempenho do modelo").run()
    figures = sorted(p.name for p in load_config().figures_dir.glob("*.png"))
    assert len(app.get("image")) == len(figures) == 8
