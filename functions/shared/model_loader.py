# -*- coding: utf-8 -*-
# Author: Daniel Maddaleno
"""Model Loader — core implementation."""
"""Lazy model loader — singleton pattern for Azure Functions cold starts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_model: Any | None = None


def get_model() -> Any:
    """Return the cached model, loading from disk on first call.

    The model path is read from the ``MODEL_PATH`` env var
    (default: ``artifacts/model.joblib``).
    """
    global _model
    if _model is not None:
        return _model

    model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    path = Path(model_path)

    if not path.exists():
        logger.warning("Model file not found at %s — using DummyModel", path)
        _model = _DummyModel()
        return _model

    import joblib
    _model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return _model


class _DummyModel:
    """Fallback predictor for local development and testing."""

    def predict(self, X):
        import numpy as np
        return np.random.rand(X.shape[0])

    def predict_proba(self, X):
        import numpy as np
        p = np.random.rand(X.shape[0])
        return np.column_stack([1 - p, p])
