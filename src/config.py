"""Central configuration and column definitions.

Loads ``config.yaml`` and is the single source of truth for feature names.

Two feature sets ("variants") are defined:

- ``early_warning`` (primary): admission data + 1st semester. The prediction is
  made at the end of the 1st semester, while there is still time to intervene.
- ``full`` (reference): also uses the 2nd semester. Reported only as a
  performance ceiling, because 2nd-semester outcomes (e.g. zero approved units)
  largely reveal a dropout that has already happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of the ``src`` package directory.
ROOT = Path(__file__).resolve().parents[1]


def _abs(path: str) -> Path:
    """Resolve a config path (relative to the project root) to an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# --- Column groups -----------------------------------------------------------
# Nominal categoricals used by the model (one-hot encoded).
CATEGORICAL_COLUMNS = [
    "Curso",
    "QualificacaoAnterior",
]

# Personal attributes deliberately kept OUT of the model (ADR 0007): removing
# them costs ~0 ROC-AUC, and a system that flags students should not score them
# on who they are. Genero is still used to audit the model's recall by group.
SENSITIVE_COLUMNS = ["Genero", "Nacionalidade", "EstadoCivil"]
FAIRNESS_GROUP_COLUMN = "Genero"

# Already-binary (0/1) flags -- passed through as numeric.
BINARY_COLUMNS = [
    "NecessidadesEspeciais",
    "Devedor",
    "MensalidadesEmDia",
    "Bolsista",
    "International",
]

# Grades known at admission.
ADMISSION_COLUMNS = ["QualificacaoAnteriorGrau", "NotaAdmissao"]

# Macroeconomic context of the student's cohort.
MACRO_COLUMNS = ["TaxaDesemprego", "TaxaInflacao", "PIB"]

# Per-semester curricular-unit counts.
SEMESTER_COUNT_COLUMNS = {
    1: [
        "UnidadesCurriculares1SemestreCreditado",
        "UnidadesCurriculares1SemestreInscrito",
        "UnidadesCurriculares1SemestreAvaliacoes",
        "UnidadesCurriculares1SemestreAprovado",
        "UnidadesCurriculares1SemestreSemAvaliacoes",
    ],
    2: [
        "UnidadesCurriculares2SemestreCreditado",
        "UnidadesCurriculares2SemestreInscrito",
        "UnidadesCurriculares2SemestreAvaliacoes",
        "UnidadesCurriculares2SemestreAprovado",
        "UnidadesCurriculares2SemestreSemAvaliacoes",
    ],
}

# Per-semester mean grade (these carry the shifted-decimal corruption).
SEMESTER_GRADE_COLUMN = {
    1: "UnidadesCurriculares1SemestreGrau",
    2: "UnidadesCurriculares2SemestreGrau",
}
GRADE_COLUMNS = [SEMESTER_GRADE_COLUMN[1], SEMESTER_GRADE_COLUMN[2]]

# --- Variants ----------------------------------------------------------------
VARIANT_SEMESTERS = {
    "early_warning": (1,),
    "full": (1, 2),
}

VARIANT_DESCRIPTIONS = {
    "early_warning": "Fim do 1º semestre (dados de ingresso + 1º semestre)",
    "full": "Fim do 2º semestre (dados de ingresso + 1º e 2º semestres)",
}


def semesters(variant: str) -> tuple[int, ...]:
    """Semesters whose data the given variant is allowed to use."""
    if variant not in VARIANT_SEMESTERS:
        raise ValueError(
            f"Variante desconhecida: {variant!r}. Use uma de {list(VARIANT_SEMESTERS)}."
        )
    return VARIANT_SEMESTERS[variant]


def grade_columns(variant: str) -> list[str]:
    return [SEMESTER_GRADE_COLUMN[s] for s in semesters(variant)]


def count_columns(variant: str) -> list[str]:
    return [c for s in semesters(variant) for c in SEMESTER_COUNT_COLUMNS[s]]


def numeric_columns(variant: str) -> list[str]:
    """Continuous / count numerics (grades are handled separately, after repair)."""
    return ADMISSION_COLUMNS + count_columns(variant) + MACRO_COLUMNS


def feature_columns(variant: str) -> list[str]:
    """All raw input columns a variant's model expects (and the app collects)."""
    return (
        CATEGORICAL_COLUMNS
        + BINARY_COLUMNS
        + grade_columns(variant)
        + numeric_columns(variant)
    )


@dataclass(frozen=True)
class Config:
    seed: int
    paths: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any]
    explain: dict[str, Any]

    @property
    def primary_variant(self) -> str:
        return self.model["primary_variant"]

    @property
    def reference_variant(self) -> str:
        return self.model["reference_variant"]

    # Convenience accessors for absolute paths.
    @property
    def raw_data_path(self) -> Path:
        return _abs(self.paths["raw_data"])

    @property
    def model_path(self) -> Path:
        return _abs(self.paths["model"])

    @property
    def metrics_path(self) -> Path:
        return _abs(self.paths["metrics"])

    @property
    def background_path(self) -> Path:
        return _abs(self.paths["background"])

    @property
    def figures_dir(self) -> Path:
        return _abs(self.paths["figures"])


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the project configuration from ``config.yaml``."""
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(
        seed=raw["seed"],
        paths=raw["paths"],
        data=raw["data"],
        model=raw["model"],
        train=raw["train"],
        explain=raw["explain"],
    )
    # Fail fast on a typo in config.yaml.
    semesters(cfg.primary_variant)
    semesters(cfg.reference_variant)
    return cfg
