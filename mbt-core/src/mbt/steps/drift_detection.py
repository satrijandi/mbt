"""Drift detection step using Population Stability Index (PSI)."""

import pandas as pd
import numpy as np
from typing import Any, List
import logging

from mbt.steps.base import Step
from mbt.core.data import PandasFrame

logger = logging.getLogger(__name__)


class DriftDetectionStep(Step):
    """Detect distribution drift using PSI (Population Stability Index).

    Calculates PSI for each feature across time periods (typically months).
    Uses the first period as baseline and compares subsequent periods to it.

    PSI Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 ≤ PSI < 0.2: Moderate drift
    - PSI ≥ 0.2: Significant drift
    """

    def run(self, inputs: dict[str, Any], context: Any) -> dict[str, Any]:
        """Calculate drift metrics for training data.

        Returns:
            {"drift_info": DataFrame with columns: feature, period, drift_method, drift_value}
        """
        # Check if temporal analysis is enabled
        temporal_config = context.get_config("training", "evaluation", "temporal_analysis")

        if not temporal_config or not temporal_config.get("enabled"):
            logger.info("Drift detection disabled - skipping")
            return {"drift_info": None}

        train_data = inputs["train_set"]
        df = train_data.to_pandas()

        # Get partition key for temporal analysis
        partition_key = context.get_config("training", "schema", "identifiers", "partition_key")

        if not partition_key or partition_key not in df.columns:
            logger.warning(f"Drift detection requires partition_key '{partition_key}' - skipping")
            return {"drift_info": None}

        # Extract time period (default to month)
        df = df.copy()
        df['_period'] = pd.to_datetime(df[partition_key]).dt.to_period('M')

        # Get feature columns
        feature_cols = self._get_feature_columns(df, context)

        if len(feature_cols) == 0:
            logger.warning("No numeric features found for drift detection")
            return {"drift_info": None}

        # Calculate PSI for each period
        drift_results = self._calculate_period_psi(df, feature_cols)

        # Log summary
        if drift_results is not None and len(drift_results) > 0:
            high_drift = drift_results[drift_results['drift_value'] > 0.2]
            moderate_drift = drift_results[
                (drift_results['drift_value'] >= 0.1) & (drift_results['drift_value'] < 0.2)
            ]

            logger.info(f"Drift analysis complete:")
            logger.info(f"  High drift (PSI ≥ 0.2): {len(high_drift)} feature-period combinations")
            logger.info(f"  Moderate drift (0.1 ≤ PSI < 0.2): {len(moderate_drift)} feature-period combinations")

            print(f"  Drift detection:")
            print(f"    High drift: {len(high_drift)} feature-period combinations")
            print(f"    Moderate drift: {len(moderate_drift)} feature-period combinations")

        return {"drift_info": PandasFrame(drift_results) if drift_results is not None else None}

    def _get_feature_columns(self, df: pd.DataFrame, context: Any) -> List[str]:
        """Get list of numeric feature columns (exclude identifiers, target, metadata)."""
        target_col = context.get_config("training", "schema", "target", "label_column")
        primary_key = context.get_config("training", "schema", "identifiers", "primary_key")
        partition_key = context.get_config("training", "schema", "identifiers", "partition_key")
        ignored_cols = context.get_config("training", "schema", "ignored_columns", default=[])

        exclude_cols = set([target_col, primary_key, partition_key, '_period'] + ignored_cols)
        exclude_cols = {col for col in exclude_cols if col is not None}

        # Only include numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        logger.info(f"Found {len(feature_cols)} numeric features for drift detection")

        return feature_cols

    def _calculate_period_psi(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """Calculate PSI for each feature across periods."""
        periods = sorted(df['_period'].unique())

        if len(periods) < 2:
            logger.warning("Need at least 2 periods for drift detection")
            return None

        baseline_period = periods[0]
        baseline_df = df[df['_period'] == baseline_period]

        logger.info(f"Using {baseline_period} as baseline period")
        logger.info(f"Analyzing {len(periods) - 1} subsequent periods")

        drift_records = []

        for period in periods[1:]:
            period_df = df[df['_period'] == period]

            for feature in feature_cols:
                # Calculate PSI
                psi = self._calculate_psi(
                    baseline_df[feature].dropna(),
                    period_df[feature].dropna()
                )

                if not np.isnan(psi):
                    drift_records.append({
                        'feature': feature,
                        'period': str(period),
                        'drift_method': 'psi',
                        'drift_value': round(psi, 4)
                    })

        return pd.DataFrame(drift_records)

    @staticmethod
    def _calculate_psi(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index.

        PSI = Σ (current% - baseline%) * ln(current% / baseline%)

        Args:
            baseline: Baseline distribution (typically first time period)
            current: Current distribution to compare
            bins: Number of bins for discretization

        Returns:
            PSI value (float). Higher values indicate more drift.
        """
        if len(baseline) == 0 or len(current) == 0:
            return np.nan

        # Create bins based on baseline distribution
        try:
            _, bin_edges = np.histogram(baseline, bins=bins)
        except (ValueError, TypeError):
            # Handle case where histogram fails (e.g., all values identical)
            return np.nan

        # Ensure bin edges cover the full range of both distributions
        bin_edges[0] = min(bin_edges[0], baseline.min(), current.min()) - 1e-6
        bin_edges[-1] = max(bin_edges[-1], baseline.max(), current.max()) + 1e-6

        # Count observations in each bin
        baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
        current_counts = np.histogram(current, bins=bin_edges)[0]

        # Calculate percentages (add small epsilon to avoid division by zero)
        epsilon = 1e-5
        baseline_pct = (baseline_counts + epsilon) / (baseline_counts.sum() + epsilon * bins)
        current_pct = (current_counts + epsilon) / (current_counts.sum() + epsilon * bins)

        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return psi
