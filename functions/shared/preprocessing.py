# -*- coding: utf-8 -*-
"""Feature preprocessing pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Columns expected by the model (order matters)
FEATURE_COLS = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "usage_frequency",
    "contract_length",
]

_scaler: StandardScaler | None = None


def _get_scaler() -> StandardScaler:
    global _scaler
    if _scaler is None:
        _scaler = StandardScaler()
    return _scaler


def preprocess(df: pd.DataFrame, fit: bool = False) -> np.ndarray:
    """Clean and transform a DataFrame into a model-ready numpy array.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.  Must contain all ``FEATURE_COLS``.
    fit : bool
        If ``True``, fit the scaler on this batch (training mode).

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    logger.info("Preprocessing %d rows", len(df))

    # Select and order columns
    X = df[FEATURE_COLS].copy()

    # Handle missing values
    for col in FEATURE_COLS:
        if X[col].isna().any():
            median = X[col].median()
            X[col] = X[col].fillna(median)
            logger.warning("Filled %d NaN in '%s' with median %.2f",
                           X[col].isna().sum(), col, median)

    # Clip outliers (IQR-based)
    for col in FEATURE_COLS:
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        clipped = X[col].clip(lower, upper)
        n_clipped = (X[col] != clipped).sum()
        if n_clipped > 0:
            logger.info("Clipped %d outliers in '%s'", n_clipped, col)
        X[col] = clipped

    # Scale
    scaler = _get_scaler()
    if fit:
        return scaler.fit_transform(X.values)
    return scaler.transform(X.values) if hasattr(scaler, "mean_") else X.values
