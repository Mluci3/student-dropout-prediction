"""Data loading, target binarization and train/test splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config, feature_columns, load_config

# Name of the engineered binary target column (1 = dropout).
TARGET = "dropout"


def load_data(cfg: Config | None = None, path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Excel dataset and drop exact duplicate rows."""
    cfg = cfg or load_config()
    src = Path(path) if path else cfg.raw_data_path
    df = pd.read_excel(src)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def binarize_target(df: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Add a binary ``dropout`` column: 1 for the positive label, 0 otherwise.

    The raw ``Target`` has three classes (Graduado / Desistente / Matriculado).
    We frame the task as "will the student drop out?", so only ``Desistente``
    becomes the positive class and both other outcomes are negatives.
    """
    cfg = cfg or load_config()
    target_col = cfg.data["target_column"]
    positive = cfg.data["positive_label"]
    out = df.copy()
    out[TARGET] = (out[target_col] == positive).astype(int)
    return out


def split_xy(df: pd.DataFrame, variant: str):
    """Return ``(X, y)`` with only the variant's feature columns and the target."""
    X = df[feature_columns(variant)].copy()
    y = df[TARGET].copy()
    return X, y


def make_split(df: pd.DataFrame, variant: str, cfg: Config | None = None):
    """Stratified train/test split on the binary target.

    The row assignment depends only on the target and the seed, so every
    variant gets exactly the same train/test students.

    Returns ``(X_train, X_test, y_train, y_test)``.
    """
    cfg = cfg or load_config()
    X, y = split_xy(df, variant)
    return train_test_split(
        X,
        y,
        test_size=cfg.data["test_size"],
        random_state=cfg.seed,
        stratify=y,
    )
