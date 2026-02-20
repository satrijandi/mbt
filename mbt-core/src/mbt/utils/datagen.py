"""Synthetic data generation for typical DS pipeline pattern."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataGenConfig:
    """Configuration for synthetic data generation."""

    # Customer population
    num_customers: int = 10000

    # Feature dimensions
    num_features_a: int = 1000
    num_features_b: int = 1000

    # Date ranges
    start_date: str = "2025-01-01"
    end_date: Optional[str] = None  # Defaults to today

    # Sampling
    daily_samples: int = 10000  # Samples per day

    # Data quality parameters
    missing_rate: float = 0.15  # 15% missing values
    constant_feature_rate: float = 0.05  # 5% constant features
    high_correlation_pairs: int = 50  # Number of highly correlated feature pairs

    # Churn parameters
    churn_rate: float = 0.2  # 20% churn rate
    churn_lookback_months: int = 3  # Churn definition window (label lag)

    # Output paths
    output_dir: str = "./sample_data"

    # Random seed
    seed: int = 42


class TypicalDSDataGenerator:
    """Generate synthetic data for typical DS pipeline pattern.

    Generates 4 CSV files:
    - label_table.csv: customer_id, snapshot_date, is_churn
    - features_table_a.csv: customer_id, snapshot_date, features_a_0001..features_a_NNNN
    - features_table_b.csv: customer_id, snapshot_date, features_b_0001..features_b_NNNN
    - customers_to_score.csv: customer_id, snapshot_date (monthly, for serving)
    """

    def __init__(self, config: DataGenConfig):
        """Initialize generator with configuration."""
        self.config = config
        np.random.seed(config.seed)

        # Parse dates
        self.start_date = pd.to_datetime(config.start_date)
        self.end_date = pd.to_datetime(config.end_date) if config.end_date else pd.Timestamp.now()

        # Generate customer IDs
        self.customer_ids = [f"CUST_{i:06d}" for i in range(1, config.num_customers + 1)]

        logger.info(f"Initialized data generator:")
        logger.info(f"  Customers: {config.num_customers}")
        logger.info(f"  Features A: {config.num_features_a}")
        logger.info(f"  Features B: {config.num_features_b}")
        logger.info(f"  Date range: {self.start_date.date()} to {self.end_date.date()}")

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all tables and return as dictionary.

        Returns:
            {
                'label_table': DataFrame,
                'features_table_a': DataFrame,
                'features_table_b': DataFrame,
                'customers_to_score': DataFrame
            }
        """
        print(f"\nGenerating synthetic data from {self.start_date.date()} to {self.end_date.date()}...")

        # Generate date range (daily)
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')

        # Generate features tables (full date range)
        print("  Generating features table A...")
        features_a = self._generate_features_table(
            date_range,
            self.config.num_features_a,
            prefix='features_a'
        )

        print("  Generating features table B...")
        features_b = self._generate_features_table(
            date_range,
            self.config.num_features_b,
            prefix='features_b'
        )

        # Generate labels (limited by churn definition lag)
        print("  Generating label table...")
        label_cutoff = self.end_date - pd.DateOffset(months=self.config.churn_lookback_months)
        label_date_range = pd.date_range(self.start_date, label_cutoff, freq='D')
        labels = self._generate_label_table(label_date_range)

        # Generate customers to score (monthly)
        print("  Generating customers to score...")
        score_start = self.start_date
        score_end = self.end_date.replace(day=1)  # First day of current month
        score_date_range = pd.date_range(score_start, score_end, freq='MS')  # Month start
        customers_to_score = self._generate_scoring_table(score_date_range)

        print(f"\n  ✓ Generated {len(labels):,} label rows")
        print(f"  ✓ Generated {len(features_a):,} feature A rows")
        print(f"  ✓ Generated {len(features_b):,} feature B rows")
        print(f"  ✓ Generated {len(customers_to_score):,} scoring rows")

        return {
            'label_table': labels,
            'features_table_a': features_a,
            'features_table_b': features_b,
            'customers_to_score': customers_to_score
        }

    def _sample_daily_customers(self, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Sample customers for each date (unique per day to avoid duplicate join keys)."""
        records = []
        for date in date_range:
            sampled_customers = np.random.choice(
                self.customer_ids,
                size=min(self.config.daily_samples, len(self.customer_ids)),
                replace=False
            )
            for customer_id in sampled_customers:
                records.append({
                    'customer_id': customer_id,
                    'snapshot_date': date.date()
                })
        return pd.DataFrame(records)

    def _generate_features_table(
        self,
        date_range: pd.DatetimeIndex,
        num_features: int,
        prefix: str
    ) -> pd.DataFrame:
        """Generate features table with realistic data quality issues."""
        # Base dataframe with customer-date combinations
        df = self._sample_daily_customers(date_range)

        # Generate feature names
        feature_cols = [f"{prefix}_{i:04d}" for i in range(1, num_features + 1)]

        # Determine which features are constant
        num_constant = int(num_features * self.config.constant_feature_rate)
        constant_indices = set(np.random.choice(num_features, num_constant, replace=False))

        # Generate features batch by batch (memory efficient)
        batch_size = 100
        for batch_start in range(0, num_features, batch_size):
            batch_end = min(batch_start + batch_size, num_features)
            batch_cols = feature_cols[batch_start:batch_end]

            for i, col in enumerate(batch_cols):
                feature_idx = batch_start + i
                if feature_idx in constant_indices:
                    # Constant feature
                    df[col] = 1.0
                else:
                    # Normal distribution
                    df[col] = np.random.randn(len(df)) * 10 + 50

        # Add high correlation pairs
        if self.config.high_correlation_pairs > 0:
            for _ in range(min(self.config.high_correlation_pairs, num_features // 2)):
                idx1, idx2 = np.random.choice(num_features, 2, replace=False)
                col1, col2 = feature_cols[idx1], feature_cols[idx2]
                # Make col2 highly correlated with col1 (correlation ~0.95)
                if col1 in df.columns and col2 in df.columns:
                    df[col2] = df[col1] * 0.95 + np.random.randn(len(df)) * 0.5

        # Inject missing values
        for col in feature_cols:
            if col in df.columns:
                mask = np.random.rand(len(df)) < self.config.missing_rate
                df.loc[mask, col] = np.nan

        # Round to 2 decimal places
        df[feature_cols] = df[feature_cols].round(2)

        return df

    def _generate_label_table(self, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate label table with churn labels."""
        df = self._sample_daily_customers(date_range)

        # Generate churn labels (binary)
        # Add customer-level stickiness (customers with lower ID churn less)
        customer_churn_prob = {}
        for cust_id in self.customer_ids:
            cust_idx = int(cust_id.split('_')[1])
            # Lower IDs = lower churn probability
            base_prob = self.config.churn_rate
            prob = base_prob * (0.5 + (cust_idx / len(self.customer_ids)))
            customer_churn_prob[cust_id] = min(prob, 0.8)

        df['is_churn'] = df['customer_id'].apply(
            lambda cid: int(np.random.rand() < customer_churn_prob.get(cid, self.config.churn_rate))
        )

        return df

    def _generate_scoring_table(self, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate customers to score (monthly snapshots)."""
        records = []
        for date in date_range:
            # Sample subset of customers for scoring each month
            num_to_score = min(self.config.daily_samples, len(self.customer_ids))
            sampled = np.random.choice(self.customer_ids, num_to_score, replace=False)
            for customer_id in sampled:
                records.append({
                    'customer_id': customer_id,
                    'snapshot_date': date.date()
                })
        return pd.DataFrame(records)

    def save_to_csv(self, output_dir: Optional[str] = None):
        """Generate and save all tables to CSV files.

        Args:
            output_dir: Directory to save files (defaults to config.output_dir)
        """
        output_path = Path(output_dir or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        tables = self.generate_all()

        print(f"\nSaving to {output_path}...")
        for name, df in tables.items():
            file_path = output_path / f"{name}.csv"
            df.to_csv(file_path, index=False)

            # Calculate file size
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file_path.name} ({len(df):,} rows, {df.shape[1]} columns, {size_mb:.1f} MB)")

        print(f"\n✓ All data saved to {output_path}/")

        return output_path
