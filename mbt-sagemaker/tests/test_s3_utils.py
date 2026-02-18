"""Tests for S3 upload utilities — header inclusion for Autopilot."""

import pytest
from unittest.mock import MagicMock, call
import pandas as pd

from mbt_sagemaker.s3_utils import upload_dataframe_to_s3


@pytest.fixture
def mock_session():
    session = MagicMock()
    return session


@pytest.fixture
def sample_df():
    return pd.DataFrame({"target": [0, 1], "feat1": [10, 20], "feat2": [30, 40]})


class TestUploadWithHeader:
    def test_no_header_by_default(self, mock_session, sample_df):
        """Built-in algorithms: default should be no header."""
        upload_dataframe_to_s3(
            df=sample_df,
            bucket="test-bucket",
            prefix="data/",
            session=mock_session,
        )

        put_call = mock_session.client.return_value.put_object
        put_call.assert_called_once()

        body = put_call.call_args[1]["Body"]
        csv_text = body.decode("utf-8") if isinstance(body, bytes) else body
        first_line = csv_text.strip().split("\n")[0]

        # No header: first line should be data, not column names
        assert "target" not in first_line
        assert "feat1" not in first_line

    def test_header_included_when_requested(self, mock_session, sample_df):
        """Autopilot: include_header=True should produce CSV with column names."""
        upload_dataframe_to_s3(
            df=sample_df,
            bucket="test-bucket",
            prefix="data/",
            session=mock_session,
            include_header=True,
        )

        put_call = mock_session.client.return_value.put_object
        body = put_call.call_args[1]["Body"]
        csv_text = body.decode("utf-8") if isinstance(body, bytes) else body
        first_line = csv_text.strip().split("\n")[0]

        assert "target" in first_line
        assert "feat1" in first_line
        assert "feat2" in first_line

    def test_returns_s3_uri(self, mock_session, sample_df):
        uri = upload_dataframe_to_s3(
            df=sample_df,
            bucket="my-bucket",
            prefix="prefix/",
            session=mock_session,
        )

        assert uri.startswith("s3://my-bucket/prefix/")
        assert uri.endswith(".csv")
