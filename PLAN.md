# Implementation Plan: Typical DS Pipeline Support

## Context

This plan enhances MBT to support the "typical data science pipeline" pattern commonly found in production ML systems. The implementation adds:

1. **Synthetic data generation** via `mbt init --template typical-ds-pipeline`
2. **Multi-table feature joins** with fan-out protection
3. **Temporal windowing** for train/test splitting based on execution dates
4. **Drift detection** using Population Stability Index (PSI)
5. **LGBM-based feature selection** with importance thresholds

### Problem

Currently, the integration test pipeline (training_churn_model_v1.yaml) works but lacks support for common production patterns:

- **Single table only** - No support for joining multiple feature tables
- **Random splitting** - No temporal windowing based on execution_date
- **No drift monitoring** - Missing PSI calculation for distribution shifts
- **Limited feature selection** - Variance/correlation only, no model-based selection

The typical_ds_pipeline/README.md describes a realistic scenario:
- 3 tables (1 label + 2 feature tables with 1000+ features each)
- 10,000 rows per day across 13+ months
- Join on composite key: [customer_id, snapshot_date]
- Temporal windows: 12 months train, 1 month test, with configurable gaps
- Label lag (e.g., 3-month churn definition creates unlabeled recent data)
- Monthly model retraining cadence

### Goal

Enable data scientists to generate realistic example projects and run production-like pipelines with:
- `mbt init my_project --template typical-ds-pipeline` (generates 4 CSV files + working pipeline)
- Full pipeline execution with multi-table joins, temporal splitting, drift detection
- Example demonstrating 2000+ features, realistic data quality issues, monthly evaluation

---

## Implementation Plan

### Phase 1: Schema Extensions (Foundation)

#### 1.1 Add feature_tables Support

**File:** `mbt-core/src/mbt/config/schema.py`

Add to `DataSourceConfig` class (currently line 33-38):

```python
class FeatureTableConfig(BaseModel):
    """Feature table specification for multi-table joins."""

    table: str = Field(..., description="Feature table name")
    join_key: list[str] = Field(
        default=["customer_id", "snapshot_date"],
        description="Columns to join on (typically primary_key + partition_key)"
    )
    join_type: str = Field(
        default="left",
        description="Join type: inner, left, right, outer"
    )
    fan_out_check: bool = Field(
        default=True,
        description="Raise error if join increases row count unexpectedly"
    )


class DataSourceConfig(BaseModel):
    """Data source specification - where to load data from."""

    label_table: str = Field(..., description="Primary table with labels")
    feature_tables: Optional[list[FeatureTableConfig]] = Field(
        default=None,
        description="Additional feature tables to join"
    )
    data_windows: Optional["DataWindowsConfig"] = Field(
        default=None,
        description="Temporal windowing configuration"
    )
```

#### 1.2 Add data_windows Configuration

**File:** `mbt-core/src/mbt/config/schema.py` (add new models)

```python
class DataWindowsConfig(BaseModel):
    """Temporal windowing configuration for train/test splitting.

    Supports two modes:
    - relative: Windows shift with execution_date (production use)
    - absolute: Fixed date ranges (reproducible experiments)
    """

    logic: str = Field(
        default="relative",
        description="Window mode: 'relative' or 'absolute'"
    )
    unit_type: str = Field(
        default="months",
        description="Time unit: days, weeks, months, quarters, years"
    )
    windows: "WindowsSpec" = Field(
        ...,
        description="Train/test window specifications"
    )


class WindowsSpec(BaseModel):
    """Window specifications for train/test splitting."""

    test_lookback_units: int = Field(
        ...,
        description="How many units of data to use for test set (from latest available)"
    )
    train_gap_units: int = Field(
        default=0,
        description="Gap between train and test sets (0 = no gap)"
    )
    train_lookback_units: int = Field(
        ...,
        description="How many units of historical data for training"
    )

    # For absolute mode (optional)
    train_start_date: Optional[str] = Field(None, description="Fixed train start (YYYY-MM-DD)")
    train_end_date: Optional[str] = Field(None, description="Fixed train end (YYYY-MM-DD)")
    test_start_date: Optional[str] = Field(None, description="Fixed test start (YYYY-MM-DD)")
    test_end_date: Optional[str] = Field(None, description="Fixed test end (YYYY-MM-DD)")
```

#### 1.3 Add Preprocessing Configuration

**File:** `mbt-core/src/mbt/config/schema.py` (add to TrainingConfig)

```python
class PreprocessingConfig(BaseModel):
    """Data preprocessing configuration."""

    remove_high_missing: Optional[dict] = Field(
        default=None,
        description="Remove columns with high missing rate. Config: {enabled: true, threshold: 0.95}"
    )
    remove_constant: Optional[dict] = Field(
        default=None,
        description="Remove constant features. Config: {enabled: true}"
    )
    imputation: Optional[dict] = Field(
        default=None,
        description="Missing value imputation. Config: {enabled: true, strategy: 'median'|'mean'|'mode'}"
    )


class TrainingConfig(BaseModel):
    """Training pipeline configuration."""

    data_source: DataSourceConfig
    schema: SchemaConfig
    preprocessing: Optional[PreprocessingConfig] = Field(default=None)
    # ... existing fields (transformations, model_training, etc.)
```

#### 1.4 Add Drift Detection Configuration

**File:** `mbt-core/src/mbt/config/schema.py` (add to EvaluationConfig)

```python
class EvaluationConfig(BaseModel):
    """Model evaluation configuration."""

    primary_metric: str = Field("roc_auc", description="Primary metric for model selection")
    additional_metrics: list[str] = Field(default_factory=list)
    generate_plots: bool = Field(True, description="Generate evaluation plots")

    temporal_analysis: Optional[dict] = Field(
        default=None,
        description="Temporal drift analysis. Config: {enabled: true, partition_key: 'snapshot_date', metrics: ['roc_auc', 'psi']}"
    )
```

---

### Phase 2: Multi-Table Join Support

#### 2.1 Update Load Data Step

**File:** `mbt-core/src/mbt/steps/load_data.py`

Modify to handle feature_tables configuration:

```python
class LoadDataStep(BaseStep):
    def run(self, inputs: dict[str, Any], context: RunContext) -> dict[str, Any]:
        # Get data source config
        data_source_config = context.get_config("data_source")

        # Load label table (existing logic)
        label_table = data_source_config["label_table"]
        connector = context.get_data_connector()
        df = connector.read_table(label_table).to_pandas()

        logger.info(f"Loaded label table '{label_table}': {len(df)} rows, {len(df.columns)} columns")

        # Load and join feature tables if configured
        feature_tables = data_source_config.get("feature_tables")
        if feature_tables:
            df = self._join_feature_tables(df, feature_tables, connector, context)

        return {"raw_data": PandasFrame(df)}

    def _join_feature_tables(
        self,
        label_df: pd.DataFrame,
        feature_tables: list[dict],
        connector: Any,
        context: RunContext
    ) -> pd.DataFrame:
        """Join multiple feature tables with label table."""
        result_df = label_df.copy()
        initial_rows = len(result_df)

        for ft_config in feature_tables:
            table_name = ft_config["table"]
            join_key = ft_config.get("join_key", ["customer_id", "snapshot_date"])
            join_type = ft_config.get("join_type", "left")
            fan_out_check = ft_config.get("fan_out_check", True)

            # Load feature table
            feature_df = connector.read_table(table_name).to_pandas()
            logger.info(f"Loaded feature table '{table_name}': {len(feature_df)} rows, {len(feature_df.columns)} columns")

            # Perform join
            result_df = result_df.merge(
                feature_df,
                on=join_key,
                how=join_type,
                suffixes=("", f"_{table_name}")
            )

            # Fan-out detection
            if fan_out_check and len(result_df) > initial_rows * 1.1:
                raise ValueError(
                    f"Join with '{table_name}' caused unexpected row increase: "
                    f"{initial_rows} → {len(result_df)}. Check for duplicate keys."
                )

            logger.info(f"After joining '{table_name}': {len(result_df)} rows, {len(result_df.columns)} columns")

        return result_df
```

---

### Phase 3: Temporal Windowing Implementation

#### 3.1 Add Window Calculation Utility

**File:** `mbt-core/src/mbt/utils/temporal.py` (NEW FILE)

```python
"""Temporal windowing utilities for train/test splitting."""

from datetime import datetime, timedelta
from typing import Tuple
import pandas as pd


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
    ) -> dict:
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
    ) -> dict:
        """Calculate windows relative to execution_date."""
        unit_type = config.get("unit_type", "months")
        windows = config["windows"]

        test_lookback = windows["test_lookback_units"]
        train_gap = windows.get("train_gap_units", 0)
        train_lookback = windows["train_lookback_units"]

        # Use available_data_end as the latest possible date
        # (accounts for label lag - e.g., can't have labels for last 3 months)
        latest_date = min(execution_date, available_data_end)

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
    def _calculate_absolute_windows(config: dict) -> dict:
        """Calculate fixed windows from absolute dates."""
        windows = config["windows"]

        return {
            "train_start": pd.to_datetime(windows["train_start_date"]),
            "train_end": pd.to_datetime(windows["train_end_date"]),
            "test_start": pd.to_datetime(windows["test_start_date"]),
            "test_end": pd.to_datetime(windows["test_end_date"])
        }

    @staticmethod
    def _subtract_units(date: datetime, units: int, unit_type: str) -> datetime:
        """Subtract time units from a date."""
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
```

#### 3.2 Update Split Data Step

**File:** `mbt-core/src/mbt/steps/split_data.py`

Replace current simple splitting with temporal logic:

```python
from mbt.utils.temporal import WindowCalculator

class SplitDataStep(BaseStep):
    def run(self, inputs: dict[str, Any], context: RunContext) -> dict[str, Any]:
        raw_data = inputs["raw_data"]
        df = raw_data.to_pandas()

        # Get data_windows config
        data_windows_config = context.get_config("data_source", "data_windows")

        if data_windows_config:
            # Temporal windowing
            train_df, test_df = self._temporal_split(df, data_windows_config, context)
        else:
            # Fallback to simple random split
            train_df, test_df = self._simple_split(df, context)

        logger.info(f"Split: {len(train_df)} train, {len(test_df)} test")

        return {
            "train_data": PandasFrame(train_df),
            "test_data": PandasFrame(test_df)
        }

    def _temporal_split(
        self,
        df: pd.DataFrame,
        data_windows_config: dict,
        context: RunContext
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data using temporal windows."""
        # Get partition key (e.g., snapshot_date)
        partition_key = context.get_config("schema", "identifiers", "partition_key")

        if not partition_key:
            raise ValueError("Temporal splitting requires partition_key in schema.identifiers")

        # Convert partition_key to datetime
        df[partition_key] = pd.to_datetime(df[partition_key])

        # Get latest available date in data (for label lag handling)
        available_data_end = df[partition_key].max()

        # Calculate windows
        windows = WindowCalculator.calculate_windows(
            execution_date=context.execution_date,
            data_windows_config=data_windows_config,
            available_data_end=available_data_end
        )

        logger.info(f"Temporal windows calculated:")
        logger.info(f"  Train: {windows['train_start'].date()} to {windows['train_end'].date()}")
        logger.info(f"  Test:  {windows['test_start'].date()} to {windows['test_end'].date()}")

        # Filter data by windows
        train_df = df[
            (df[partition_key] >= windows['train_start']) &
            (df[partition_key] < windows['train_end'])
        ].copy()

        test_df = df[
            (df[partition_key] >= windows['test_start']) &
            (df[partition_key] < windows['test_end'])
        ].copy()

        if len(train_df) == 0:
            raise ValueError(f"No training data in window {windows['train_start']} to {windows['train_end']}")

        if len(test_df) == 0:
            raise ValueError(f"No test data in window {windows['test_start']} to {windows['test_end']}")

        return train_df, test_df

    def _simple_split(self, df: pd.DataFrame, context: RunContext) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fallback: simple random split (existing implementation)."""
        from sklearn.model_selection import train_test_split

        test_size = context.get_config("split", "test_size", default=0.2)
        stratify_col = context.get_config("schema", "target", "label_column")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[stratify_col] if stratify_col in df else None,
            random_state=42
        )

        return train_df, test_df
```

---

### Phase 4: Drift Detection (PSI)

#### 4.1 Create Drift Detection Step

**File:** `mbt-core/src/mbt/steps/drift_detection.py` (NEW FILE)

```python
"""Drift detection step using Population Stability Index (PSI)."""

import pandas as pd
import numpy as np
from typing import Any, Dict
import logging

from mbt.core.step import BaseStep
from mbt.core.context import RunContext

logger = logging.getLogger(__name__)


class DriftDetectionStep(BaseStep):
    """Detect distribution drift using PSI (Population Stability Index)."""

    def run(self, inputs: dict[str, Any], context: RunContext) -> dict[str, Any]:
        train_data = inputs["train_data"].to_pandas()

        # Get partition key for month-over-month analysis
        partition_key = context.get_config("schema", "identifiers", "partition_key")

        if not partition_key or partition_key not in train_data.columns:
            logger.warning("Drift detection requires partition_key - skipping")
            return {"drift_info": None}

        # Extract month from partition key
        train_data = train_data.copy()
        train_data['_month'] = pd.to_datetime(train_data[partition_key]).dt.to_period('M')

        # Get feature columns (exclude target, identifiers, metadata)
        feature_cols = self._get_feature_columns(train_data, context)

        # Calculate PSI for each month vs. first month (baseline)
        drift_results = self._calculate_monthly_psi(train_data, feature_cols)

        # Log summary
        if drift_results:
            high_drift = drift_results[drift_results['drift_value'] > 0.2]
            logger.info(f"Drift analysis: {len(high_drift)} features with PSI > 0.2 (high drift)")

        return {"drift_info": drift_results}

    def _get_feature_columns(self, df: pd.DataFrame, context: RunContext) -> list[str]:
        """Get list of feature columns (exclude identifiers, target, metadata)."""
        target_col = context.get_config("schema", "target", "label_column")
        primary_key = context.get_config("schema", "identifiers", "primary_key")
        partition_key = context.get_config("schema", "identifiers", "partition_key")
        ignored_cols = context.get_config("schema", "ignored_columns", default=[])

        exclude_cols = set([target_col, primary_key, partition_key, '_month'] + ignored_cols)

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def _calculate_monthly_psi(
        self,
        df: pd.DataFrame,
        feature_cols: list[str]
    ) -> pd.DataFrame:
        """Calculate PSI for each feature across months."""
        months = sorted(df['_month'].unique())

        if len(months) < 2:
            logger.warning("Need at least 2 months for drift detection")
            return None

        baseline_month = months[0]
        baseline_df = df[df['_month'] == baseline_month]

        drift_records = []

        for month in months[1:]:
            month_df = df[df['_month'] == month]

            for feature in feature_cols:
                # Skip non-numeric features
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    continue

                # Calculate PSI
                psi = self._calculate_psi(
                    baseline_df[feature].dropna(),
                    month_df[feature].dropna()
                )

                drift_records.append({
                    'feature': feature,
                    'month': str(month),
                    'drift_method': 'psi',
                    'drift_value': psi
                })

        return pd.DataFrame(drift_records)

    @staticmethod
    def _calculate_psi(baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index.

        PSI = Σ (current% - baseline%) * ln(current% / baseline%)

        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.2: Moderate drift
        - PSI ≥ 0.2: Significant drift
        """
        if len(baseline) == 0 or len(current) == 0:
            return np.nan

        # Create bins based on baseline distribution
        _, bin_edges = np.histogram(baseline, bins=bins)

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
```

#### 4.2 Register Drift Detection in Compiler

**File:** `mbt-core/src/mbt/core/compiler.py`

Add drift detection to step graph (after split_data, before train_model):

```python
# In build_step_graph method
if self.config.training.get("evaluation", {}).get("temporal_analysis", {}).get("enabled"):
    graph.add_step("drift_detection", DriftDetectionStep(), deps=["split_data"])
    train_deps.append("drift_detection")
```

---

### Phase 5: LGBM Feature Selection

#### 5.1 Extend Feature Selection Step

**File:** `mbt-core/src/mbt/steps/feature_selection.py`

Add LGBM importance method (requires `lightgbm` dependency):

```python
class FeatureSelectionStep(BaseStep):
    def run(self, inputs: dict[str, Any], context: RunContext) -> dict[str, Any]:
        # ... existing code ...

        # Apply each feature selection method
        for method_config in methods:
            method_name = method_config["name"]

            if method_name == "variance_threshold":
                selected_features = self._variance_threshold(X_train, method_config)
            elif method_name == "correlation":
                selected_features = self._correlation_filter(X_train, method_config)
            elif method_name == "mutual_info":
                selected_features = self._mutual_info(X_train, y_train, method_config, problem_type)
            elif method_name == "lgbm_importance":
                selected_features = self._lgbm_importance(X_train, y_train, method_config, problem_type)
            else:
                logger.warning(f"Unknown feature selection method: {method_name}")
                continue

            # ... rest of existing logic ...

    def _lgbm_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        config: dict,
        problem_type: str
    ) -> list[str]:
        """Select features using LightGBM importance.

        Trains a LightGBM model and selects features based on cumulative importance.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("lightgbm not installed - skipping lgbm_importance")
            return X.columns.tolist()

        threshold = config.get("threshold", 0.95)  # Cumulative importance threshold

        # Train LightGBM
        if problem_type == "binary_classification" or problem_type == "multiclass_classification":
            model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        else:
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

        model.fit(X, y)

        # Get feature importances
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Calculate cumulative importance
        importances['cumulative_importance'] = importances['importance'].cumsum() / importances['importance'].sum()

        # Select features up to threshold
        selected = importances[importances['cumulative_importance'] <= threshold]['feature'].tolist()

        logger.info(f"LGBM importance: Selected {len(selected)}/{len(X.columns)} features (threshold={threshold})")

        return selected
```

Add `lightgbm` to `mbt-core/pyproject.toml` dependencies (optional):

```toml
[project.optional-dependencies]
feature-selection = ["lightgbm>=4.0.0"]
```

---

### Phase 6: Data Generation Module

#### 6.1 Create Data Generator

**File:** `mbt-core/src/mbt/utils/datagen.py` (NEW FILE)

(Use the comprehensive implementation from the Plan agent's design - full code from earlier)

Key features:
- Generate 4 tables: label, features_a, features_b, customers_to_score
- 10,000 daily samples with configurable customer count
- Inject realistic data quality issues (15% missing, 5% constant features)
- Memory-efficient chunked generation for 1000+ feature columns
- Configurable date ranges and label lag

#### 6.2 Create Pipeline Templates

**File:** `mbt-core/src/mbt/utils/pipeline_templates.py` (NEW FILE)

Functions to generate:
- `generate_typical_pipeline_yaml(project_name: str) -> str`
- `generate_profiles_yaml(project_name: str) -> str`

Use YAML that fully leverages the new features (feature_tables, data_windows, drift detection, lgbm selection).

---

### Phase 7: CLI Integration

#### 7.1 Enhance mbt init Command

**File:** `mbt-core/src/mbt/cli.py`

Add `--template` parameter and data generation options:

```python
@app.command()
def init(
    project_name: str = typer.Argument(..., help="Name of the project to create"),
    template: str = typer.Option(
        "basic",
        "--template",
        "-t",
        help="Project template: basic, typical-ds-pipeline"
    ),
    # Data generation options
    num_customers: int = typer.Option(10000, "--num-customers"),
    num_features_a: int = typer.Option(1000, "--features-a"),
    num_features_b: int = typer.Option(1000, "--features-b"),
    start_date: str = typer.Option("2025-01-01", "--start-date"),
    end_date: str = typer.Option(None, "--end-date"),
    daily_samples: int = typer.Option(10000, "--daily-samples"),
    seed: int = typer.Option(42, "--seed"),
):
    """Initialize a new MBT project."""
    # ... existing directory creation ...

    if template == "typical-ds-pipeline":
        _generate_typical_ds_pipeline(project_path, project_name, {
            'num_customers': num_customers,
            'num_features_a': num_features_a,
            'num_features_b': num_features_b,
            'start_date': start_date,
            'end_date': end_date,
            'daily_samples': daily_samples,
            'seed': seed,
        })
```

---

### Phase 8: Testing

#### 8.1 Unit Tests

**File:** `mbt-core/tests/utils/test_datagen.py` (NEW FILE)
- Test DataGenConfig validation
- Test individual table generation
- Test data quality injection

**File:** `mbt-core/tests/utils/test_temporal.py` (NEW FILE)
- Test WindowCalculator with relative mode
- Test WindowCalculator with absolute mode
- Test edge cases (label lag, invalid configs)

**File:** `mbt-core/tests/steps/test_drift_detection.py` (NEW FILE)
- Test PSI calculation
- Test monthly drift analysis
- Test with single month (should skip gracefully)

#### 8.2 Integration Tests

**File:** `mbt-core/tests/integration/test_typical_pipeline.py` (NEW FILE)

End-to-end test:
1. Generate project via `mbt init test_project --template typical-ds-pipeline`
2. Verify all files created
3. Compile pipeline
4. Run pipeline
5. Verify outputs (metrics, drift_info, model artifacts)

---

## Critical Files Summary

### Files to Create (8 new files)
1. `mbt-core/src/mbt/utils/datagen.py` - Synthetic data generator
2. `mbt-core/src/mbt/utils/temporal.py` - Window calculation utilities
3. `mbt-core/src/mbt/utils/pipeline_templates.py` - YAML generation
4. `mbt-core/src/mbt/steps/drift_detection.py` - PSI drift detection step
5. `mbt-core/tests/utils/test_datagen.py` - Data generator tests
6. `mbt-core/tests/utils/test_temporal.py` - Temporal utils tests
7. `mbt-core/tests/steps/test_drift_detection.py` - Drift detection tests
8. `mbt-core/tests/integration/test_typical_pipeline.py` - E2E integration test

### Files to Modify (5 files)
1. `mbt-core/src/mbt/config/schema.py` - Add 4 new models (FeatureTableConfig, DataWindowsConfig, WindowsSpec, PreprocessingConfig)
2. `mbt-core/src/mbt/cli.py` - Add --template flag and data generation logic
3. `mbt-core/src/mbt/steps/load_data.py` - Add multi-table join support
4. `mbt-core/src/mbt/steps/split_data.py` - Add temporal windowing logic
5. `mbt-core/src/mbt/steps/feature_selection.py` - Add LGBM importance method
6. `mbt-core/src/mbt/core/compiler.py` - Register drift_detection step
7. `mbt-core/pyproject.toml` - Add optional lightgbm dependency

---

## Verification Steps

After implementation, verify with:

```bash
# 1. Generate example project
mbt init my_churn_project --template typical-ds-pipeline

# 2. Verify files created
cd my_churn_project
ls -lh sample_data/  # Should have 4 CSV files (~100MB+ total)

# 3. Compile pipeline
mbt compile my_churn_project_training_v1

# 4. Check manifest
cat target/manifest.json | jq '.steps | keys'
# Should include: load_data, split_data, drift_detection, feature_selection, train_model, evaluate, log_run

# 5. Run pipeline
mbt run --select my_churn_project_training_v1

# 6. Verify outputs
ls local_artifacts/run_*/
# Should contain: metrics.json, drift_info.csv, model artifacts, plots

# 7. Check drift detection results
head local_artifacts/run_*/drift_info.csv
# Should show: features, month, drift_method, drift_value

# 8. Verify temporal splitting worked
cat logs/split_data.log | grep "Temporal windows"
# Should show calculated train/test date ranges

# 9. Run tests
uv run pytest mbt-core/tests/integration/test_typical_pipeline.py -v
```

---

## Dependencies

New Python packages required:
- `lightgbm>=4.0.0` (optional, for LGBM feature selection)
- No other new dependencies (numpy, pandas, scikit-learn already present)

---

## Implementation Order

1. **Schema extensions** (Phase 1) - Foundation for all other features
2. **Temporal windowing** (Phase 3) - Core feature, no external dependencies
3. **Multi-table joins** (Phase 2) - Builds on schema, straightforward
4. **Data generator** (Phase 6) - Can develop in parallel with features
5. **Drift detection** (Phase 4) - Independent step, can add anytime
6. **LGBM selection** (Phase 5) - Optional feature, minimal changes
7. **CLI integration** (Phase 7) - Ties everything together
8. **Testing** (Phase 8) - Continuous throughout, final validation at end

---

## Success Criteria

✅ User can run: `mbt init demo --template typical-ds-pipeline`
✅ Generated project has 4 CSV tables with realistic data (10k customers, 2000+ features, 13+ months)
✅ Generated pipeline YAML compiles without errors
✅ Pipeline runs end-to-end with multi-table joins, temporal splitting, drift detection
✅ Outputs include monthly AUC metrics, PSI drift analysis, selected features list
✅ Integration test passes with full pipeline execution
✅ Documentation updated with typical-ds-pipeline example
