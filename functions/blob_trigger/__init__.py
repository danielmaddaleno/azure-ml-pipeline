# -*- coding: utf-8 -*-
"""Azure Function — BlobTrigger for ML inference pipeline.

Fires when a CSV is uploaded to the ``input/`` container.
Reads the file, validates schema, runs preprocessing + prediction,
and writes results to the ``output/`` container.
"""

from __future__ import annotations

import io
import logging

import azure.functions as func
import pandas as pd

from shared.model_loader import get_model
from shared.preprocessing import preprocess
from shared.schemas import validate_input

logger = logging.getLogger(__name__)


def main(inputBlob: func.InputStream, outputBlob: func.Out[str]) -> None:
    blob_name = inputBlob.name
    logger.info("Processing blob: %s (%d bytes)", blob_name, inputBlob.length)

    # 1. Read CSV
    raw = inputBlob.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # 2. Validate schema
    df = validate_input(df)

    # 3. Preprocess
    X = preprocess(df)

    # 4. Predict
    model = get_model()
    predictions = model.predict(X)
    df["prediction"] = predictions
    logger.info("Generated %d predictions", len(predictions))

    # 5. Write results
    output_csv = df.to_csv(index=False)
    outputBlob.set(output_csv)
    logger.info("Wrote predictions to output container")

__all__: list[str] = []

__version__ = "1.0.0"
