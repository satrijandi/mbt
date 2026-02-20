"""Temporal windowing utilities for train/test splitting."""

from datetime import datetime, timedelta
from typing import Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class WindowCalculator:
    """Calculate train/test date windows based on configuration."""

    UNIT_MAPPING = {
        "days": "D",
        "weeks": "W",
        "months": "M",
        "quarters": "Q",
        "years": "Y"
    }

    @staticmethod
    def calculate_windows(
        execution_date: datetime,
        data_windows_config: dict,
        available_data_end: datetime
    ) -> Dict[str, datetime]:
        """Calculate train/test windows from config and execution date.

        Args:
            execution_date: Reference date for relative windows
            data_windows_config: Configuration dict with logic, unit_type, windows
            available_data_end: Latest date with available data (for label lag handling)

        Returns:
            {
                'train_start': datetime,
                'train_end': datetime,
                'test_start': datetime,
                'test_end': datetime
            }
        """
        logic = data_windows_config.get("logic", "relative")

        if logic == "absolute":
            return WindowCalculator._calculate_absolute_windows(data_windows_config)
        else:
            return WindowCalculator._calculate_relative_windows(
                execution_date,
                data_windows_config,
                available_data_end
            )

    @staticmethod
    def _calculate_relative_windows(
        execution_date: datetime,
        config: dict,
        available_data_end: datetime
    ) -> Dict[str, datetime]:
        """Calculate windows relative to execution_date."""
        unit_type = config.get("unit_type", "months")
        windows = config["windows"]

        test_lookback = windows["test_lookback_units"]
        train_gap = windows.get("train_gap_units", 0)
        train_lookback = windows["train_lookback_units"]

        # Use available_data_end as the latest possible date
        # (accounts for label lag - e.g., can't have labels for last 3 months)
        latest_date = min(execution_date, available_data_end)

        logger.info(f"Calculating relative windows: execution_date={execution_date.date()}, available_data_end={available_data_end.date()}")

        # Calculate test window (most recent data)
        test_end = latest_date
        test_start = WindowCalculator._subtract_units(test_end, test_lookback, unit_type)

        # Calculate train window (before test, with optional gap)
        train_end = WindowCalculator._subtract_units(test_start, train_gap, unit_type)
        train_start = WindowCalculator._subtract_units(train_end, train_lookback, unit_type)

        return {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end
        }

    @staticmethod
    def _calculate_absolute_windows(config: dict) -> Dict[str, datetime]:
        """Calculate fixed windows from absolute dates."""
        windows = config["windows"]

        logger.info("Using absolute (fixed) date windows")

        return {
            "train_start": pd.to_datetime(windows["train_start_date"]),
            "train_end": pd.to_datetime(windows["train_end_date"]),
            "test_start": pd.to_datetime(windows["test_start_date"]),
            "test_end": pd.to_datetime(windows["test_end_date"])
        }

    @staticmethod
    def _subtract_units(date: datetime, units: int, unit_type: str) -> datetime:
        """Subtract time units from a date.

        Args:
            date: Starting date
            units: Number of units to subtract
            unit_type: Type of unit (days, weeks, months, quarters, years)

        Returns:
            Resulting datetime after subtraction
        """
        if units == 0:
            return date

        if unit_type == "days":
            return date - timedelta(days=units)
        elif unit_type == "weeks":
            return date - timedelta(weeks=units)
        elif unit_type == "months":
            return date - pd.DateOffset(months=units)
        elif unit_type == "quarters":
            return date - pd.DateOffset(months=units * 3)
        elif unit_type == "years":
            return date - pd.DateOffset(years=units)
        else:
            raise ValueError(f"Unsupported unit_type: {unit_type}")
