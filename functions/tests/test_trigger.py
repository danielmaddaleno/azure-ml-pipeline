# -*- coding: utf-8 -*-
"""Test Trigger — automated test suite."""
"""Tests for the BlobTrigger function using mocks."""

from unittest.mock import MagicMock, patch
import pytest


def _make_input_blob(csv_content: str, name: str = "input/test.csv"):
    blob = MagicMock()
    blob.name = name
    blob.length = len(csv_content)
    blob.read.return_value = csv_content.encode("utf-8")
    return blob


SAMPLE_CSV = (
    "tenure_months,monthly_spend,support_tickets,usage_frequency,contract_length\n"
    "12,99.0,1,0.8,12\n"
    "24,150.0,0,0.5,24\n"
)


class TestBlobTrigger:
    @patch("shared.model_loader.get_model")
    def test_trigger_produces_output(self, mock_model):
        import numpy as np
        mock_model.return_value = MagicMock(
            predict=MagicMock(return_value=np.array([0.1, 0.9]))
        )

        from blob_trigger import main

        input_blob = _make_input_blob(SAMPLE_CSV)
        output_blob = MagicMock()

        main(input_blob, output_blob)

        output_blob.set.assert_called_once()
        written = output_blob.set.call_args[0][0]
        assert "prediction" in written
        assert "0.1" in written or "0.9" in written
