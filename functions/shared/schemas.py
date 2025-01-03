"""Pydantic-based input validation for incoming CSVs."""

from __future__ import annotations

import logging

import pandas as pd
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "usage_frequency",
    "contract_length",
]


class InputRow(BaseModel):
    """Schema for a single input row."""

    tenure_months: float
    monthly_spend: float
    support_tickets: int
    usage_frequency: float
    contract_length: int

    @field_validator("tenure_months", "monthly_spend", "usage_frequency")
    @classmethod
    def must_be_non_negative(cls, v: float, info) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be >= 0, got {v}")
        return v


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate that *df* has the required columns and coercible types.

    Drops rows that fail validation and logs warnings.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    valid_rows: list[int] = []
    for idx, row in df.iterrows():
        try:
            InputRow(**{col: row[col] for col in REQUIRED_COLUMNS})
            valid_rows.append(idx)
        except Exception as exc:
            logger.warning("Row %d failed validation: %s", idx, exc)

    dropped = len(df) - len(valid_rows)
    if dropped:
        logger.warning("Dropped %d / %d invalid rows", dropped, len(df))

    return df.loc[valid_rows].reset_index(drop=True)
