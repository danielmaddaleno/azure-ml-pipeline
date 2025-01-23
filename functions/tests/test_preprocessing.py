"""Tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from shared.preprocessing import FEATURE_COLS, preprocess


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "tenure_months": [12, 24, 6, 36, 48],
        "monthly_spend": [99.0, 150.0, 50.0, 200.0, 120.0],
        "support_tickets": [1, 0, 3, 2, 0],
        "usage_frequency": [0.8, 0.5, 0.9, 0.3, 0.7],
        "contract_length": [12, 24, 6, 12, 24],
    })


class TestPreprocess:
    def test_output_shape(self, sample_df):
        X = preprocess(sample_df)
        assert X.shape == (5, len(FEATURE_COLS))

    def test_handles_nan(self, sample_df):
        sample_df.loc[0, "monthly_spend"] = np.nan
        X = preprocess(sample_df)
        assert not np.isnan(X).any()

    def test_column_order(self, sample_df):
        # Shuffle columns
        shuffled = sample_df[list(reversed(FEATURE_COLS))]
        X = preprocess(shuffled)
        assert X.shape[1] == len(FEATURE_COLS)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"tenure_months": [1]})
        with pytest.raises(KeyError):
            preprocess(df)
