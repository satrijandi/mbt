# MBT (Model Build Tool) Implementation Plan

## Context

MBT is a declarative ML pipeline framework for tabular machine learning. The goal is to enable data scientists to define training and serving pipelines in YAML (declarative "what"), while the framework compiles these declarations into executable task graphs (imperative "how"). This is similar to how dbt works for analytics, but adapted for ML workflows with an AutoML-first approach.

**Problem being solved**: Current ML tooling requires data scientists to write extensive Python code defining DAG tasks, wiring dependencies, and managing infrastructure. MBT separates concerns: data scientists own YAML pipeline definitions, data engineers own profiles.yaml for environment configuration, and the framework handles compilation, validation, and execution.

**Key architectural principles**:
1. **Declarative YAML interface** - Data scientists declare intent, not implementation
2. **Compile before execute** - All validation happens at compile time, producing manifest.json
3. **Modular adapter ecosystem** - Core framework has zero ML/infra dependencies; everything is a plugin
4. **Two-layer separation** - DS writes YAML → compiler generates manifest → DE configures execution
5. **AutoML-first** - Strong models without hyperparameter tuning via H2O AutoML and similar frameworks

## Implementation Strategy

The implementation follows a 6-phase approach, building from a minimal viable core to a full-featured framework. Each phase delivers working, testable functionality that proves critical architectural components.

---

## Phase 1: Foundation - Minimal Viable Pipeline (Days 1-2)

**Goal**: Execute a simple end-to-end training pipeline locally with hardcoded steps.

### What to Build

#### Core Infrastructure

**CLI Framework** - [mbt-core/src/mbt/cli.py](mbt-core/src/mbt/cli.py)
- Typer-based CLI with commands: `init`, `compile`, `run`, `validate`
- `mbt init` scaffolds a new project with pyproject.toml, pipelines/, lib/, sample_data/
- `mbt compile <pipeline>` produces target/<pipeline>/manifest.json
- `mbt run --select <pipeline>` executes all steps and produces run_results.json

**Compiler (Simplified)** - [mbt-core/src/mbt/core/compiler.py](mbt-core/src/mbt/core/compiler.py)
- Phase 2 only: Schema validation via Pydantic
- Load pipeline YAML, validate against schema, produce manifest
- No base_pipeline, no !include, no plugin validation yet
- Manifest generation: merge YAML step configs into StepDefinition objects

**DAG Builder** - [mbt-core/src/mbt/core/dag.py](mbt-core/src/mbt/core/dag.py)
- Build linear DAG from hardcoded step list: load_data → split_data → train_model → evaluate
- Simple topological sort (Python graphlib)
- execution_batches field (for now: one step per batch)

**Runner** - [mbt-core/src/mbt/core/runner.py](mbt-core/src/mbt/core/runner.py)
- Execute steps in order from manifest
- Artifact passing: after each step, serialize outputs to storage; before next step, load inputs
- Use StoragePlugin for all artifact I/O
- Generate run_results.json with status, metrics, duration per step

**Data Protocol** - [mbt-core/src/mbt/core/data.py](mbt-core/src/mbt/core/data.py)
```python
class MBTFrame(Protocol):
    def to_pandas(self) -> pd.DataFrame
    def num_rows(self) -> int
    def columns(self) -> list[str]
    def schema(self) -> dict[str, str]

class PandasFrame(MBTFrame):
    # Default implementation wrapping pandas DataFrame
```

#### Schema Definition

**Pipeline YAML Schema** - [mbt-core/src/mbt/config/schema.py](mbt-core/src/mbt/config/schema.py)
- Pydantic models for schema_version, project, training sections
- Subset for Phase 1: project, data_source (label_table only), schema (target, identifiers), model_training, evaluation
- All optional/advanced sections stubbed for later phases

#### Built-in Components

**Local Storage** - [mbt-core/src/mbt/builtins/local_storage.py](mbt-core/src/mbt/builtins/local_storage.py)
- Implements StoragePlugin ABC
- put(artifact_name, data, run_id, step_name) → saves to ./local_artifacts/{run_id}/{step_name}/{name}
- get(uri) → loads from filesystem

**Local Connector** - [mbt-core/src/mbt/builtins/local_connector.py](mbt-core/src/mbt/builtins/local_connector.py)
- Implements DataConnectorPlugin ABC (stubbed for now)
- read_table(table, columns, date_range) → loads CSV/Parquet from sample_data/

**Local Executor** - [mbt-core/src/mbt/builtins/local_executor.py](mbt-core/src/mbt/builtins/local_executor.py)
- Execute steps as Python function calls (not subprocesses)
- Just imports and calls step.run(inputs, context)

#### Pipeline Steps (Hardcoded)

**Step Base Class** - [mbt-core/src/mbt/steps/base.py](mbt-core/src/mbt/steps/base.py)
```python
class Step(ABC):
    @abstractmethod
    def run(self, inputs: dict[str, Any], context: dict) -> dict[str, Any]:
        """Execute step. Returns {output_name: output_value}"""
```

**Load Data** - [mbt-core/src/mbt/steps/load_data.py](mbt-core/src/mbt/steps/load_data.py)
- Read CSV from sample_data/ using local_connector
- Return {"raw_data": PandasFrame(df)}

**Split Data** - [mbt-core/src/mbt/steps/split_data.py](mbt-core/src/mbt/steps/split_data.py)
- Simple train/test split (80/20 ratio, stratified by target)
- Temporal windowing comes in Phase 4
- Return {"train_set": ..., "test_set": ...}

**Train Model** - [mbt-core/src/mbt/steps/train_model.py](mbt-core/src/mbt/steps/train_model.py)
- **HARDCODED**: Use sklearn RandomForestClassifier
- No plugin system yet
- Return {"model": fitted_model, "train_metrics": {...}}

**Evaluate** - [mbt-core/src/mbt/steps/evaluate.py](mbt-core/src/mbt/steps/evaluate.py)
- Compute metrics: accuracy, precision, recall, f1, roc_auc (for binary classification)
- Return {"eval_metrics": {...}}

#### Example Project

**Sample Pipeline** - [examples/telecom-churn/pipelines/churn_simple_v1.yaml](examples/telecom-churn/pipelines/churn_simple_v1.yaml)
```yaml
schema_version: 1

project:
  name: churn_simple_v1
  problem_type: binary_classification
  owner: data_science

training:
  data_source:
    label_table: customers

  schema:
    target:
      label_column: churned
      classes: [0, 1]
      positive_class: 1
    identifiers:
      primary_key: customer_id

  model_training:
    framework: sklearn  # hardcoded for now
    config:
      model: RandomForestClassifier
      n_estimators: 100

  evaluation:
    primary_metric: roc_auc
    additional_metrics: [accuracy, f1, precision, recall]
```

**Sample Data** - [examples/telecom-churn/sample_data/customers.csv](examples/telecom-churn/sample_data/customers.csv)
- 500 rows with: customer_id, tenure, monthly_charges, total_charges, churned (0/1)
- Simple feature set for testing

### Success Criteria

```bash
# Initialize project
mbt init my-ml-project

# Compile pipeline
cd examples/telecom-churn
mbt compile churn_simple_v1
# ✓ Produces: target/churn_simple_v1/manifest.json

# Inspect manifest
cat target/churn_simple_v1/manifest.json
# Shows: 4 steps (load_data, split_data, train_model, evaluate)
# Each step has: plugin, config, inputs, outputs, depends_on

# Run pipeline
mbt run --select churn_simple_v1
# ✓ Executes all steps sequentially
# ✓ Produces: target/churn_simple_v1/run_results.json
# ✓ Saves artifacts: local_artifacts/run_<id>/{model, train_set, test_set, metrics}
# ✓ Console shows: step progress, metrics (ROC AUC ~0.85)
```

### What's Deliberately Simplified

- No plugin registry - steps are hardcoded imports
- No profiles.yaml - everything runs locally with defaults
- No base_pipeline or !include - composition comes in Phase 3
- No data validation - just load and run
- No MLflow integration - logging comes in Phase 2
- No normalization/encoding/feature_selection - Phase 4
- No temporal windowing - simple 80/20 split
- No serving pipeline - Phase 5

---

## Phase 2: Adapter System - Prove the Architecture (Days 3-4)

**Goal**: Replace hardcoded sklearn with a plugin registry and build 3 real adapters to validate the contract design.

### What to Build

#### Plugin System

**Plugin Registry** - [mbt-core/src/mbt/core/registry.py](mbt-core/src/mbt/core/registry.py)
- Discover adapters via Python entry_points mechanism
- get(group, name) → load and instantiate plugin
- list_installed() → show all available adapters by category
- Clear error messages when adapter not found: "No adapter 'X'. Install: pip install mbt-X"

**Contract ABCs** - [mbt-core/src/mbt/contracts/](mbt-core/src/mbt/contracts/)
```python
# framework.py
class FrameworkPlugin(ABC):
    def setup(self, config: dict) -> None: pass  # default: no-op
    def teardown(self) -> None: pass
    def health_check(self) -> bool: return True
    def supported_formats(self) -> list[str]: return ["pandas"]

    @abstractmethod
    def validate_config(self, config: dict, problem_type: str) -> None:
        """Compile-time validation. Raise ValueError on invalid params."""

    @abstractmethod
    def train(self, X_train: MBTFrame, y_train: MBTFrame, config: dict) -> Any:
        """Train and return model object."""

    @abstractmethod
    def predict(self, model: Any, X: MBTFrame) -> np.ndarray:
        """Generate predictions."""

    @abstractmethod
    def serialize(self, model: Any, path: str) -> None: ...
    @abstractmethod
    def deserialize(self, path: str) -> Any: ...

# storage.py
class StoragePlugin(ABC):
    @abstractmethod
    def put(self, artifact_name: str, data: bytes, run_id: str,
            step_name: str, metadata: dict | None = None) -> str:
        """Store artifact, return URI."""

    @abstractmethod
    def get(self, artifact_uri: str) -> bytes: ...
    @abstractmethod
    def exists(self, artifact_uri: str) -> bool: ...

# model_registry.py
class ModelRegistryPlugin(ABC):
    @abstractmethod
    def log_run(self, pipeline_name: str, metrics: dict, params: dict,
                artifacts: dict, tags: dict) -> str:
        """Log training run, return run_id."""

    @abstractmethod
    def load_model(self, run_id: str) -> Any: ...
    @abstractmethod
    def load_artifacts(self, run_id: str) -> dict: ...
```

#### Enhanced Compiler

**Phase 3: Plugin Validation** - Update [mbt-core/src/mbt/core/compiler.py](mbt-core/src/mbt/core/compiler.py)
- After schema validation, load framework plugin from registry
- Call plugin.validate_config(config, problem_type)
- Catch and report validation errors with clear context

**DAG Builder Enhancement** - Update [mbt-core/src/mbt/core/dag.py](mbt-core/src/mbt/core/dag.py)
- Steps now reference plugin entry point in manifest (e.g., "mbt_h2o.framework:H2OAutoMLFramework")
- Runner uses registry to instantiate plugins dynamically

#### Adapter Packages

**mbt-sklearn** - [mbt-sklearn/src/mbt_sklearn/framework.py](mbt-sklearn/src/mbt_sklearn/framework.py)
- Implement FrameworkPlugin for sklearn
- validate_config: check model name is valid sklearn class
- train: instantiate model class, fit, return
- Support: RandomForestClassifier, LogisticRegression, GradientBoostingClassifier
- Entry point: `[project.entry-points."mbt.frameworks"] sklearn = "mbt_sklearn.framework:SklearnFramework"`

**mbt-h2o** - [mbt-h2o/src/mbt_h2o/framework.py](mbt-h2o/src/mbt_h2o/framework.py)
- Implement FrameworkPlugin for H2O AutoML
- setup(): h2o.init()
- teardown(): h2o.cluster().shutdown()
- validate_config: check sort_metric valid for problem_type
- train: H2OAutoML with max_runtime_secs, max_models, sort_metric
- Return: aml.leader (best model)
- Entry point: `[project.entry-points."mbt.frameworks"] h2o_automl = "mbt_h2o.framework:H2OAutoMLFramework"`

**mbt-mlflow** - [mbt-mlflow/src/mbt_mlflow/registry.py](mbt-mlflow/src/mbt_mlflow/registry.py)
- Implement ModelRegistryPlugin
- log_run: create MLflow run, log metrics/params/tags, log model + artifacts, return run_id
- load_model: mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
- load_artifacts: download artifacts from run (scaler, encoder, etc.)
- Entry point: `[project.entry-points."mbt.model_registries"] mlflow = "mbt_mlflow.registry:MLflowRegistry"`

#### Updated Steps

**Train Model (Plugin-based)** - Update [mbt-core/src/mbt/steps/train_model.py](mbt-core/src/mbt/steps/train_model.py)
```python
def run(self, inputs: dict, context: dict) -> dict:
    framework_name = context["config"]["model_training"]["framework"]
    framework = context["registry"].get("mbt.frameworks", framework_name)

    framework.setup(context["config"])
    model = framework.train(
        X_train=inputs["train_X"],
        y_train=inputs["train_y"],
        config=context["config"]["model_training"]["config"]
    )
    framework.teardown()

    return {"model": model}
```

**Log Run** - [mbt-core/src/mbt/steps/log_run.py](mbt-core/src/mbt/steps/log_run.py)
- New step that logs to model registry
- Collects all metrics, artifacts (model, scaler, etc.), params, tags
- Calls registry.log_run(), returns {"mlflow_run_id": run_id}

#### Updated Example

**Full Pipeline** - [examples/telecom-churn/pipelines/churn_training_v1.yaml](examples/telecom-churn/pipelines/churn_training_v1.yaml)
```yaml
training:
  model_training:
    framework: h2o_automl  # or: sklearn
    config:
      max_runtime_secs: 600
      max_models: 10
      sort_metric: AUC
      seed: 42
```

**Project Config** - [examples/telecom-churn/pyproject.toml](examples/telecom-churn/pyproject.toml)
```toml
[project]
name = "telecom-churn-example"
dependencies = [
    "mbt-core>=0.1.0",
    "mbt-h2o>=0.1.0",
    "mbt-sklearn>=0.1.0",
    "mbt-mlflow>=0.1.0",
]
```

### Success Criteria

```bash
# Install adapters
cd examples/telecom-churn
pip install -e ../../mbt-core -e ../../mbt-sklearn -e ../../mbt-h2o -e ../../mbt-mlflow

# Discover installed adapters
mbt deps list
# Output:
#   mbt.frameworks: h2o_automl, sklearn
#   mbt.model_registries: mlflow

# Compile with validation
mbt compile churn_training_v1
# ✓ Framework plugin validates config
# ✓ Catches invalid sort_metric at compile time

# Run with H2O AutoML
mbt run --select churn_training_v1
# ✓ h2o.init() called
# ✓ H2O AutoML trains multiple models
# ✓ Best model selected by AUC
# ✓ Results logged to MLflow
# ✓ Console shows: MLflow run_id

# Switch frameworks by changing YAML
# Change: framework: sklearn, config: {model: RandomForestClassifier, n_estimators: 200}
mbt compile churn_training_v1
mbt run --select churn_training_v1
# ✓ Now uses sklearn RandomForest
# ✓ Same pipeline otherwise
# ✓ No code changes needed
```

### Validation

The adapter architecture is proven if:
1. Framework plugins are discovered via entry_points
2. Same pipeline YAML works with multiple frameworks (sklearn, H2O)
3. Compile-time validation catches framework-specific config errors
4. MLflow integration works without hardcoding

---

## Phase 3: Configuration System - Profiles and Targets (Days 5-6)

**Goal**: Add profiles.yaml for multi-environment deployments and complete the 5-phase compiler.

### What to Build

#### Configuration System

**Profiles Loader** - [mbt-core/src/mbt/config/profiles.py](mbt-core/src/mbt/config/profiles.py)
- Load profiles.yaml from project root or ~/.mbt/profiles.yaml
- Resolve target: CLI flag --target → MBT_TARGET env var → default from profiles
- Return merged config: profile-level settings + target-specific overrides

**Jinja2 Template Engine** - [mbt-core/src/mbt/config/loader.py](mbt-core/src/mbt/config/loader.py)
- Support {{ env_var('KEY') }} in profiles.yaml
- Support {{ secret('KEY') }} (resolved at runtime via secrets provider)
- Support {{ var('KEY') }} for runtime variables (--vars key=value)
- Clear errors for undefined variables

**Composition Resolver** - [mbt-core/src/mbt/core/composition.py](mbt-core/src/mbt/core/composition.py)
```python
class CompositionResolver:
    def resolve_base_pipeline(self, yaml_path: Path) -> dict:
        """Resolve base_pipeline (single-level only). Deep merge semantics."""

    def resolve_includes(self, yaml_dict: dict) -> dict:
        """Replace !include directives with file contents."""
```

**Secrets Provider** - [mbt-core/src/mbt/contracts/secrets.py](mbt-core/src/mbt/contracts/secrets.py)
```python
class SecretsPlugin(ABC):
    @abstractmethod
    def get_secret(self, key: str) -> str: ...
    @abstractmethod
    def validate_access(self) -> bool: ...
```

**Environment Secrets** - [mbt-core/src/mbt/builtins/env_secrets.py](mbt-core/src/mbt/builtins/env_secrets.py)
- Default secrets provider: reads from environment variables
- Entry point: `[project.entry-points."mbt.secrets"] env = "mbt.builtins.env_secrets:EnvSecretsProvider"`

#### Enhanced Compiler (Now All 5 Phases)

Update [mbt-core/src/mbt/core/compiler.py](mbt-core/src/mbt/core/compiler.py):

```python
class Compiler:
    def compile(self, pipeline_name: str, target: str) -> Manifest:
        # Phase 1: Resolution
        yaml_dict = self._resolve_composition(pipeline_name)

        # Phase 2: Schema validation
        pipeline = self._validate_schema(yaml_dict)

        # Phase 3: Plugin validation
        self._validate_plugins(pipeline)

        # Phase 4: DAG assembly
        dag = self._build_dag(pipeline)

        # Phase 5: Manifest generation
        manifest = self._generate_manifest(pipeline, dag, target)
        return manifest
```

#### Profiles Configuration

**profiles.yaml Structure** - [examples/telecom-churn/profiles.yaml](examples/telecom-churn/profiles.yaml)
```yaml
telecom-churn:
  target: dev

  mlflow:
    tracking_uri: "{{ env_var('MLFLOW_TRACKING_URI') }}"

  secrets:
    provider: env

  outputs:
    dev:
      executor:
        type: local
      storage:
        type: local
        config:
          base_path: ./local_artifacts
      data_connector:
        type: local_file
        config:
          data_path: ./sample_data
      mlflow:
        tracking_uri: "sqlite:///mlruns.db"

    staging:
      executor:
        type: kubernetes  # config only, execution stubbed for now
        config:
          namespace: ml-staging
      storage:
        type: s3
        config:
          bucket: myorg-ml-staging
          region: us-east-1
      data_connector:
        type: snowflake
        config:
          account: "{{ secret('SNOWFLAKE_ACCOUNT') }}"
          warehouse: ML_WH_SMALL
      mlflow:
        tracking_uri: "https://mlflow.staging.myorg.com"
```

#### Pipeline Composition

**Base Pipeline** - [examples/telecom-churn/pipelines/_base_churn.yaml](examples/telecom-churn/pipelines/_base_churn.yaml)
```yaml
schema_version: 1

project:
  experiment_name: churn_prediction
  owner: customer_analytics_team
  problem_type: binary_classification
  tags: [churn, telecom]

training:
  data_source:
    label_table: customer_churn_features

  schema: !include ../includes/telecom_schema.yaml

  evaluation:
    primary_metric: roc_auc
    additional_metrics: [accuracy, f1, precision, recall]
    generate_plots: true
```

**Derived Pipeline** - [examples/telecom-churn/pipelines/churn_h2o_v1.yaml](examples/telecom-churn/pipelines/churn_h2o_v1.yaml)
```yaml
schema_version: 1
base_pipeline: _base_churn

project:
  name: churn_h2o_v1
  tags: [churn, telecom, h2o]

training:
  model_training:
    framework: h2o_automl
    config:
      max_runtime_secs: 3600
      max_models: 20
      sort_metric: AUC
```

**YAML Fragment** - [examples/telecom-churn/includes/telecom_schema.yaml](examples/telecom-churn/includes/telecom_schema.yaml)
```yaml
target:
  label_column: churned
  classes: [0, 1]
  positive_class: 1
identifiers:
  primary_key: customer_id
  partition_key: snapshot_date
ignored_columns:
  - customer_name
  - email
  - phone_number
```

### Success Criteria

```bash
# Compile for different targets
mbt compile churn_h2o_v1 --target dev
# ✓ Uses: local executor, local storage, sqlite MLflow

mbt compile churn_h2o_v1 --target staging
# ✓ Uses: K8s executor config, S3 storage, remote MLflow
# ✓ Manifest includes target-specific config

# Show resolved YAML
mbt validate churn_h2o_v1 --show-resolved
# ✓ Output: fully merged YAML
# ✓ Includes: all fields from _base_churn.yaml
# ✓ Includes: content from telecom_schema.yaml
# ✓ Overrides: project.name, project.tags, model_training section

# Multiple pipelines share base
mbt compile churn_h2o_v1    # extends _base_churn, uses H2O
mbt compile churn_xgb_v1    # extends _base_churn, uses XGBoost
# ✓ Both inherit data_source, schema, evaluation from base
# ✓ Only differ in model_training section
```

---

## Phase 4: Complete Pipeline Steps - Transformations and Validation (Days 7-8)

**Goal**: Implement all missing pipeline steps: validation, joins, normalization, encoding, feature selection.

### What to Build

#### New Steps

**Join Tables** - [mbt-core/src/mbt/steps/join_tables.py](mbt-core/src/mbt/steps/join_tables.py)
- Multi-table joins with explicit join_key, join_type
- Validate: join keys exist in both tables
- Fan-out check: fail if join produces > 1x rows when fan_out_check: true

**Validate Data** - [mbt-core/src/mbt/steps/validate_data.py](mbt-core/src/mbt/steps/validate_data.py)
- Built-in checks: null_threshold, value_range, expected_columns, unique_key, type_check
- Custom validators: dynamically import from lib/
- Failure handling: fail (halt), warn (log + continue), skip_row (remove bad rows)

**Normalize** - [mbt-core/src/mbt/steps/normalize.py](mbt-core/src/mbt/steps/normalize.py)
- Methods: standard_scaler, min_max_scaler, robust_scaler
- Fit on train, transform train and test
- Return: {"normalized_train": ..., "normalized_test": ..., "scaler_artifact": ...}

**Encode** - [mbt-core/src/mbt/steps/encode.py](mbt-core/src/mbt/steps/encode.py)
- One-hot encoding, label encoding
- Fit on train, transform train and test

**Feature Selection** - [mbt-core/src/mbt/steps/feature_selection.py](mbt-core/src/mbt/steps/feature_selection.py)
- Methods: lgbm_importance (cumulative threshold), correlation filtering
- Fit on train, select features, apply to train and test

#### Enhanced Split Data

**Temporal Windowing** - Update [mbt-core/src/mbt/steps/split_data.py](mbt-core/src/mbt/steps/split_data.py)
```python
def run(self, inputs: dict, context: dict) -> dict:
    data_windows = context["config"]["training"]["data_source"]["data_windows"]
    execution_date = context["execution_date"]

    # Calculate label_window (offset from execution_date)
    label_end = execution_date + timedelta(months=data_windows["label_window"]["offset"])

    # Calculate train_window (duration before label_window)
    train_start = label_end - timedelta(months=data_windows["train_window"]["duration"])
    train_end = label_end

    # Filter data by time windows
    # Stratified split for test set
```

#### Step Registry

**Conditional DAG Builder** - Update [mbt-core/src/mbt/core/dag.py](mbt-core/src/mbt/core/dag.py)
```python
TRAINING_STEP_REGISTRY = [
    StepReg("load_data", LoadDataStep, condition=always_true),
    StepReg("join_tables", JoinTablesStep,
            condition=lambda cfg: len(cfg["training"]["data_source"].get("feature_tables", [])) > 1),
    StepReg("validate_data", ValidateDataStep,
            condition=lambda cfg: "validation" in cfg["training"]),
    StepReg("split_data", SplitDataStep, condition=always_true),
    StepReg("normalize", NormalizeStep,
            condition=lambda cfg: cfg["training"]["transformations"]["normalization"]["enabled"]),
    StepReg("select_features", FeatureSelectionStep,
            condition=lambda cfg: cfg["training"]["feature_selection"]["enabled"]),
    StepReg("train_model", TrainModelStep, condition=always_true),
    StepReg("evaluate", EvaluateStep, condition=always_true),
    StepReg("log_to_mlflow", LogRunStep, condition=always_true),
]

def build_dag(self, pipeline_config: dict) -> DAG:
    steps = [reg.step_class for reg in TRAINING_STEP_REGISTRY
             if reg.condition(pipeline_config)]
    return self._wire_dependencies(steps)
```

#### Hook Functions

**Custom Transform Support** - [examples/telecom-churn/lib/custom_transforms.py](examples/telecom-churn/lib/custom_transforms.py)
```python
def compute_rfm(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Compute recency, frequency, monetary features."""
    df["recency"] = (context["execution_date"] - df["last_purchase"]).dt.days
    df["frequency"] = df.groupby("customer_id")["order_id"].transform("count")
    df["monetary"] = df.groupby("customer_id")["amount"].transform("sum")
    return df
```

**Custom Validators** - [examples/telecom-churn/lib/custom_validators.py](examples/telecom-churn/lib/custom_validators.py)
```python
def check_label_distribution(df: pd.DataFrame, context: dict) -> tuple[bool, str]:
    label_col = context["schema"]["target"]["label_column"]
    positive_rate = df[label_col].mean()
    if positive_rate < 0.01 or positive_rate > 0.99:
        return False, f"Label imbalance extreme: {positive_rate:.4f}"
    return True, f"Label positive rate: {positive_rate:.4f}"
```

#### Updated Example

**Full Pipeline YAML** - [examples/telecom-churn/pipelines/churn_training_v1.yaml](examples/telecom-churn/pipelines/churn_training_v1.yaml)
```yaml
training:
  data_source:
    label_table: customer_churn_features
    feature_tables:
      - table: customer_churn_features
      - table: customer_demographics
        join_key: customer_id
        join_type: left
    data_windows:
      label_window: {offset: -1, unit: month}
      train_window: {duration: 12, unit: month}

  validation:
    on_failure: fail
    checks:
      - type: null_threshold
        columns: [churned, customer_id]
        max_null_pct: 0.0
      - type: value_range
        column: monthly_charges
        min: 0
        max: 500
      - type: custom
        function: lib.custom_validators.check_label_distribution

  transformations:
    normalization:
      enabled: true
      method: standard_scaler
    custom_transforms:
      - lib.custom_transforms.compute_rfm

  feature_selection:
    enabled: true
    methods:
      - name: lgbm_importance
        threshold: 0.95
```

### Success Criteria

```bash
# Full pipeline with all steps
mbt compile churn_training_v1
# ✓ Manifest shows: load_data → join_tables → validate_data → split_data →
#                    normalize → select_features → train_model → evaluate → log

# Toggle sections
# Set: feature_selection.enabled: false
mbt compile churn_training_v1
# ✓ Manifest skips select_features step

# Data validation catches issues
# Corrupt sample data: add nulls to 'churned' column
mbt run --select churn_training_v1
# ✓ Fails at validate_data step
# ✓ Clear error: "null_threshold check failed: churned has 5% nulls (max 0%)"

# Custom transforms execute
mbt run --select churn_training_v1
# ✓ compute_rfm called during transformation
# ✓ Output data includes recency, frequency, monetary columns

# Temporal windowing
mbt run --select churn_training_v1 --vars execution_date=2026-03-01
# ✓ Label window: February 2026
# ✓ Train window: February 2025 → January 2026 (12 months)
```

---

## Phase 5: Serving Pipeline and Orchestrator Integration (Days 9-10)

**Goal**: Implement serving pipelines with run_id-based artifact resolution and Airflow DAG generation.

### What to Build

#### Serving Steps

**Load Model** - [mbt-core/src/mbt/steps/load_model.py](mbt-core/src/mbt/steps/load_model.py)
```python
def run(self, inputs: dict, context: dict) -> dict:
    run_id = context["config"]["serving"]["model_source"]["run_id"]
    registry = context["registry"].get("mbt.model_registries", "mlflow")

    model = registry.load_model(run_id)
    artifacts = registry.load_artifacts(run_id)  # scaler, encoder, selector

    return {"model": model, **artifacts}
```

**Apply Transforms** - [mbt-core/src/mbt/steps/apply_transforms.py](mbt-core/src/mbt/steps/apply_transforms.py)
- Load scoring data
- Apply scaler.transform() (if exists)
- Apply encoder.transform() (if exists)
- Apply feature selector (if exists)
- Ensure column order matches training

**Predict** - [mbt-core/src/mbt/steps/predict.py](mbt-core/src/mbt/steps/predict.py)
- Use framework plugin to generate predictions
- Return: {"predictions": ..., "prediction_proba": ...}

**Publish** - [mbt-core/src/mbt/steps/publish.py](mbt-core/src/mbt/steps/publish.py)
- Write predictions via output_writer plugin
- Columns: customer_id, prediction, prediction_proba, execution_date

#### Artifact Snapshot System

**Compile-time Artifact Fetching** - Update [mbt-core/src/mbt/core/compiler.py](mbt-core/src/mbt/core/compiler.py)
```python
# Phase 5: Manifest generation (enhanced)
if pipeline_type == "serving" and config["serving"]["model_source"].get("artifact_snapshot"):
    run_id = config["serving"]["model_source"]["run_id"]
    registry = self.registry.get("mbt.model_registries", "mlflow")

    # Fetch all artifacts at compile time
    artifacts = registry.download_artifacts(run_id, output_dir=f"target/{pipeline_name}/artifacts/")
    manifest.artifact_snapshot = artifacts
```

#### Orchestrator Adapter

**Airflow DAG Builder** - [mbt-airflow/src/mbt_airflow/dag_builder.py](mbt-airflow/src/mbt_airflow/dag_builder.py)
```python
class AirflowDagBuilder(OrchestratorPlugin):
    def generate_dag_file(self, manifest_path: str, output_path: str, **kwargs):
        manifest = self._load_manifest(manifest_path)
        pipeline_name = manifest["metadata"]["pipeline_name"]

        dag_code = f'''
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

dag = DAG(
    dag_id="ml_{pipeline_name}",
    schedule="{kwargs.get("schedule", "@daily")}",
    start_date=datetime(2026, 1, 1),
    catchup=False,
)

tasks = {{}}
'''

        # Generate one task per step
        for step_name, step_def in manifest["steps"].items():
            dag_code += f'''
tasks["{step_name}"] = BashOperator(
    task_id="{step_name}",
    bash_command="mbt exec --pipeline {pipeline_name} --step {step_name} --target {manifest["metadata"]["target"]}",
    dag=dag,
)
'''

        # Wire dependencies
        for step_name, parents in manifest["dag"]["parent_map"].items():
            for parent in parents:
                dag_code += f'tasks["{parent}"] >> tasks["{step_name}"]\n'

        with open(output_path, "w") as f:
            f.write(dag_code)
```

**OrchestratorPlugin Contract** - [mbt-core/src/mbt/contracts/orchestrator.py](mbt-core/src/mbt/contracts/orchestrator.py)
```python
class OrchestratorPlugin(ABC):
    @abstractmethod
    def generate_dag_file(self, manifest_path: str, output_path: str,
                          schedule: str | None, target: str, **kwargs) -> None:
        """Generate orchestrator-native DAG file from manifest."""
```

#### CLI Commands

**Step Execution** - Update [mbt-core/src/mbt/cli.py](mbt-core/src/mbt/cli.py)
```python
@app.command()
def exec(pipeline: str, step: str, target: str):
    """Execute a single step (used by orchestrators)."""
    manifest = load_manifest(f"target/{pipeline}/manifest.json")
    runner = Runner(manifest, target)
    runner.run_step(step)
```

**DAG Generation** - Update [mbt-core/src/mbt/cli.py](mbt-core/src/mbt/cli.py)
```python
@app.command()
def generate_dags(target: str, output: str = "./generated_dags/"):
    """Generate orchestrator DAG files for all compiled pipelines."""
    for manifest_path in Path("target/").glob("*/manifest.json"):
        orchestrator = get_orchestrator_from_profile(target)
        orchestrator.generate_dag_file(
            manifest_path=str(manifest_path),
            output_path=f"{output}/ml_{manifest_path.parent.name}.py",
            target=target
        )
```

#### Serving Pipeline Example

**Serving YAML** - [examples/telecom-churn/pipelines/churn_serving_v1.yaml](examples/telecom-churn/pipelines/churn_serving_v1.yaml)
```yaml
schema_version: 1

project:
  name: churn_serving_v1
  experiment_name: churn_prediction
  problem_type: binary_classification
  tags: [churn, telecom, serving]

deployment:
  mode: batch
  cadence: daily

serving:
  model_source:
    registry: mlflow
    run_id: "{{ var('run_id') }}"
    artifact_snapshot: true
    fallback_run_id: "{{ var('fallback') }}"

  data_source:
    scoring_table: active_customers
    data_windows:
      lookback_window:
        offset: -1
        unit: day

  output:
    destination: local_file
    table: predictions.csv
    columns:
      prediction: churn_probability
      label: churn_predicted
      threshold: 0.5
```

### Success Criteria

```bash
# Train a model
mbt run --select churn_training_v1
# ✓ MLflow run_id: abc123def456

# Compile serving pipeline with run_id
mbt compile churn_serving_v1 --vars run_id=abc123def456
# ✓ Compiler fetches model + artifacts from MLflow
# ✓ Manifest includes artifact snapshot

# Run serving pipeline
mbt run --select churn_serving_v1 --vars run_id=abc123def456
# ✓ Loads model, scaler, feature selector
# ✓ Loads scoring data
# ✓ Applies transforms in correct order
# ✓ Generates predictions
# ✓ Writes to predictions.csv

# Generate Airflow DAG
mbt generate-dags --target prod --output ./dags/
# ✓ Produces: dags/ml_churn_training_v1.py
# ✓ DAG file contains: one BashOperator per step
# ✓ Dependencies wired via >> operator

# Verify DAG structure
cat dags/ml_churn_training_v1.py
# ✓ Shows: load_data >> validate_data >> split_data >> ... >> log_to_mlflow
# ✓ Each task runs: mbt exec --pipeline churn_training_v1 --step <step> --target prod
```

---

## Phase 6: Testing Framework and Observability (Day 11)

**Goal**: Add testing, observability, and production-readiness features.

### What to Build

#### Testing System

**DAG Assertions** - [mbt-core/src/mbt/testing/assertions.py](mbt-core/src/mbt/testing/assertions.py)
```python
class DagAssertion(BaseModel):
    type: str
    # step_exists, step_absent, step_order, step_count, resource_limit

def run_assertions(manifest: dict, assertions: list[DagAssertion]) -> list[TestResult]:
    results = []
    for assertion in assertions:
        if assertion.type == "step_exists":
            passed = assertion.step in manifest["steps"]
        elif assertion.type == "step_order":
            # Check that 'before' appears before 'after' in execution_batches
        # ...
    return results
```

**Test Fixtures** - [mbt-core/src/mbt/testing/fixtures.py](mbt-core/src/mbt/testing/fixtures.py)
```python
class MockMBTFrame:
    """Mock for adapter unit tests."""
    def __init__(self, data: dict):
        self._df = pd.DataFrame(data)
    def to_pandas(self): return self._df
    def num_rows(self): return len(self._df)

class MockStoragePlugin:
    """In-memory storage for testing."""
    def __init__(self):
        self._store = {}
    def put(self, name, data, run_id, step_name, metadata=None):
        uri = f"mock://{run_id}/{step_name}/{name}"
        self._store[uri] = data
        return uri
```

**Test Command** - Update [mbt-core/src/mbt/cli.py](mbt-core/src/mbt/cli.py)
```python
@app.command()
def test(pipeline: str | None = None):
    """Run DAG assertion tests."""
    test_files = find_test_files(pipeline)
    for test_file in test_files:
        test_spec = load_yaml(test_file)
        manifest = compile_pipeline(test_spec["pipeline"])
        results = run_assertions(manifest, test_spec["assertions"])
        print_results(results)
```

#### Observability

**Structured Logging** - [mbt-core/src/mbt/observability/logging.py](mbt-core/src/mbt/observability/logging.py)
```python
class StructuredLogger:
    def log_step_start(self, pipeline: str, step: str, run_id: str, target: str):
        log_json({
            "ts": datetime.utcnow().isoformat(),
            "level": "INFO",
            "pipeline": pipeline,
            "step": step,
            "run_id": run_id,
            "target": target,
            "event": "step_started",
        })
```

**Metrics Emission** - [mbt-core/src/mbt/observability/metrics.py](mbt-core/src/mbt/observability/metrics.py)
```python
class MetricsEmitter:
    def emit_step_duration(self, pipeline: str, step: str, duration: float):
        # Send to StatsD or Prometheus
        metric_name = f"mbt.pipelines.{pipeline}.{step}.duration_seconds"
        self.client.gauge(metric_name, duration)
```

**Enhanced Runner** - Update [mbt-core/src/mbt/core/runner.py](mbt-core/src/mbt/core/runner.py)
```python
def run_step(self, step_name: str):
    logger.log_step_start(...)
    start = time.time()

    try:
        result = step.run(inputs, context)
        duration = time.time() - start

        logger.log_step_complete(step_name, duration)
        metrics.emit_step_duration(pipeline, step_name, duration)
        return result
    except Exception as e:
        logger.log_step_failure(step_name, str(e))
        metrics.emit_step_status(pipeline, step_name, status=1)
        raise
```

#### CLI Enhancements

**DAG Visualization** - Update [mbt-core/src/mbt/cli.py](mbt-core/src/mbt/cli.py)
```python
@app.command()
def dag(pipeline: str, mermaid: bool = False):
    """Visualize pipeline DAG."""
    manifest = load_manifest(f"target/{pipeline}/manifest.json")
    if mermaid:
        print(generate_mermaid_diagram(manifest))
    else:
        print(generate_ascii_dag(manifest))
```

**Debug Command** - Update [mbt-core/src/mbt/cli.py](mbt-core/src/mbt/cli.py)
```python
@app.command()
def debug(target: str):
    """Validate connections and secrets for a target."""
    profile = load_profile(target)

    # Test secrets provider
    secrets = get_secrets_provider(profile)
    print(f"Secrets provider: {secrets.validate_access()}")

    # Test MLflow connection
    mlflow_uri = profile["mlflow"]["tracking_uri"]
    print(f"MLflow connection: {test_mlflow_connection(mlflow_uri)}")

    # Test storage backend
    storage = get_storage_plugin(profile)
    print(f"Storage: {storage.health_check()}")
```

**Retry Command** - Update [mbt-core/src/mbt/cli.py](mbt-core/src/mbt/cli.py)
```python
@app.command()
def retry(pipeline: str, force: bool = False):
    """Re-run pipeline from failure point."""
    manifest = load_manifest(f"target/{pipeline}/manifest.json")
    last_run = load_run_results(f"target/{pipeline}/run_results.json")

    for step_name in manifest["dag"]["execution_order"]:
        step = manifest["steps"][step_name]

        if not force and step["idempotent"]:
            # Check if outputs exist
            if all_outputs_exist(step_name, last_run["run_id"]):
                print(f"Skipping {step_name} (outputs exist)")
                continue

        run_step(step_name)
```

#### Test Files

**DAG Assertions** - [examples/telecom-churn/tests/churn_training_v1.test.yaml](examples/telecom-churn/tests/churn_training_v1.test.yaml)
```yaml
pipeline: churn_training_v1

assertions:
  - type: step_exists
    step: validate_data

  - type: step_exists
    step: normalize

  - type: step_order
    before: normalize
    after: train_model

  - type: step_count
    min: 7
    max: 10

  - type: resource_limit
    step: train_model
    memory_max: "64Gi"
```

### Success Criteria

```bash
# Run DAG tests
mbt test
# ✓ Executes all .test.yaml files
# ✓ All assertions pass
# ✓ Exit code 0

# Visualize DAG
mbt dag churn_training_v1 --mermaid
# ✓ Outputs Mermaid diagram
# ✓ Shows: load_data → validate → split → normalize → select → train → evaluate → log

# Debug connections
mbt debug --target prod
# ✓ Tests secrets provider: ✓ Connected
# ✓ Tests MLflow: ✓ Reachable
# ✓ Tests S3 storage: ✓ Bucket accessible

# Retry after failure
# Simulate: train_model fails
mbt retry churn_training_v1
# ✓ Skips: load_data, validate_data, split_data, normalize (outputs exist)
# ✓ Re-runs: train_model, evaluate, log_to_mlflow

# Structured logs
mbt run --select churn_training_v1 2>&1 | jq
# ✓ JSON log lines with: ts, level, pipeline, step, run_id, event
```

---

## Critical Files to Implement

These 5 files are the foundation of the architecture and must be implemented correctly:

1. **[mbt-core/src/mbt/core/compiler.py](mbt-core/src/mbt/core/compiler.py)** - 5-phase compilation with base_pipeline resolution and plugin validation
2. **[mbt-core/src/mbt/contracts/framework.py](mbt-core/src/mbt/contracts/framework.py)** - FrameworkPlugin ABC with lifecycle hooks and MBTFrame support
3. **[mbt-core/src/mbt/core/runner.py](mbt-core/src/mbt/core/runner.py)** - Step execution engine with artifact passing via StoragePlugin
4. **[mbt-core/src/mbt/config/schema.py](mbt-core/src/mbt/config/schema.py)** - Pydantic models for pipeline YAML schema (versioned, toggleable)
5. **[mbt-h2o/src/mbt_h2o/framework.py](mbt-h2o/src/mbt_h2o/framework.py)** - H2O AutoML adapter proving the AutoML-first vision

## Project Structure

```
/workspaces/mbt/
├── mbt-core/                           # Core framework package
│   ├── src/mbt/
│   │   ├── cli.py                      # Typer CLI
│   │   ├── core/
│   │   │   ├── compiler.py             # 5-phase compilation
│   │   │   ├── dag.py                  # DAG builder with step registry
│   │   │   ├── manifest.py             # Pydantic manifest models
│   │   │   ├── runner.py               # Step execution + artifact passing
│   │   │   ├── registry.py             # Plugin discovery
│   │   │   ├── composition.py          # base_pipeline + !include
│   │   │   ├── data.py                 # MBTFrame protocol
│   │   │   └── context.py              # Runtime context
│   │   ├── contracts/                  # Adapter ABCs
│   │   │   ├── framework.py
│   │   │   ├── data_connector.py
│   │   │   ├── storage.py
│   │   │   ├── model_registry.py
│   │   │   ├── orchestrator.py
│   │   │   ├── secrets.py
│   │   │   └── ...
│   │   ├── config/
│   │   │   ├── schema.py               # Pipeline YAML Pydantic models
│   │   │   ├── profiles.py             # Profile/target resolution
│   │   │   └── loader.py               # Jinja2 template engine
│   │   ├── steps/                      # Built-in steps
│   │   │   ├── base.py
│   │   │   ├── load_data.py
│   │   │   ├── join_tables.py
│   │   │   ├── validate_data.py
│   │   │   ├── split_data.py
│   │   │   ├── normalize.py
│   │   │   ├── encode.py
│   │   │   ├── feature_selection.py
│   │   │   ├── train_model.py
│   │   │   ├── evaluate.py
│   │   │   ├── log_run.py
│   │   │   ├── load_model.py
│   │   │   ├── apply_transforms.py
│   │   │   ├── predict.py
│   │   │   └── publish.py
│   │   ├── builtins/                   # Minimal built-in adapters
│   │   │   ├── local_executor.py
│   │   │   ├── local_connector.py
│   │   │   ├── local_storage.py
│   │   │   └── env_secrets.py
│   │   ├── testing/
│   │   │   ├── fixtures.py             # Mock objects
│   │   │   └── assertions.py           # DAG assertion engine
│   │   └── observability/
│   │       ├── logging.py              # Structured logging
│   │       └── metrics.py              # StatsD/Prometheus
│   ├── tests/
│   └── pyproject.toml
│
├── mbt-sklearn/                        # scikit-learn adapter
│   ├── src/mbt_sklearn/framework.py
│   ├── tests/
│   └── pyproject.toml
│
├── mbt-h2o/                            # H2O AutoML adapter
│   ├── src/mbt_h2o/framework.py
│   ├── tests/
│   └── pyproject.toml
│
├── mbt-mlflow/                         # MLflow registry adapter
│   ├── src/mbt_mlflow/registry.py
│   ├── tests/
│   └── pyproject.toml
│
├── mbt-airflow/                        # Airflow orchestrator adapter
│   ├── src/mbt_airflow/dag_builder.py
│   ├── tests/
│   └── pyproject.toml
│
└── examples/
    └── telecom-churn/                  # Example project
        ├── pyproject.toml
        ├── profiles.yaml
        ├── pipelines/
        │   ├── _base_churn.yaml
        │   ├── churn_simple_v1.yaml    # Phase 1
        │   ├── churn_training_v1.yaml  # Full training
        │   ├── churn_h2o_v1.yaml       # H2O variant
        │   └── churn_serving_v1.yaml   # Serving
        ├── includes/
        │   └── telecom_schema.yaml
        ├── tests/
        │   └── churn_training_v1.test.yaml
        ├── lib/
        │   ├── custom_transforms.py
        │   ├── custom_validators.py
        │   └── custom_metrics.py
        └── sample_data/
            ├── customers.csv
            └── demographics.csv
```

## Verification Strategy

After each phase:

1. **Unit tests** - Test individual components in isolation
2. **Integration test** - Run full pipeline end-to-end with sample data
3. **CLI test** - Verify all CLI commands work as documented
4. **Example project** - telecom-churn example must compile and run successfully

Final validation (Phase 6):
- All DAG assertion tests pass
- `mbt debug --target prod` validates all connections
- Airflow DAG generates correctly and is valid Python
- Full training → serving workflow works with run_id resolution

## Estimated Timeline

- **Phase 1**: 2 days - Foundation
- **Phase 2**: 2 days - Adapter system
- **Phase 3**: 1-2 days - Configuration
- **Phase 4**: 1-2 days - Transformations
- **Phase 5**: 2 days - Serving + orchestration
- **Phase 6**: 1 day - Testing + observability

**Total: 9-11 days** of focused implementation

## What This Proves

By Phase 6, we will have demonstrated:

1. ✅ **Declarative pipeline definition** - DS writes YAML, framework compiles to DAG
2. ✅ **Adapter ecosystem viability** - Multiple adapters (sklearn, H2O, MLflow, Airflow) work without mbt-core changes
3. ✅ **Compile-time validation** - Invalid configs caught before execution
4. ✅ **Multi-environment deployment** - Same YAML works across dev/staging/prod
5. ✅ **Serving eliminates schema drift** - run_id-based artifact resolution
6. ✅ **Orchestrator decoupling** - Airflow DAG generated from manifest.json
7. ✅ **Pipeline composition** - base_pipeline and !include prevent duplication
8. ✅ **AutoML-first approach** - H2O AutoML produces strong models without tuning

This proves the core value proposition: **Data scientists declare what they want in YAML, MBT compiles it to an executable DAG, and adapters make it run anywhere.**
