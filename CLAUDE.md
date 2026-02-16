# CLAUDE.md - MBT Project Guide

## Project Overview

MBT (Model Build Tool) is a declarative ML pipeline framework for tabular machine learning, inspired by dbt. It enables data scientists to define ML workflows in YAML while data engineers manage execution environments through profiles. The framework compiles YAML declarations into executable DAGs that can run locally or on any orchestrator (Airflow, Prefect, Dagster).

**Tagline:** "DBT for DS and DE"

## Architecture Principles

### Core Design Philosophy

1. **Declarative over Imperative** - Data scientists declare *what* they want (data, transforms, models, metrics), not *how* to execute it
2. **Compile Before Execute** - Every run first compiles YAML into `manifest.json`, catching errors before computation starts
3. **Separation of Concerns**:
   - Data Scientists: Own `pipelines/*.yaml` and `lib/` custom transforms
   - Data Engineers: Own `profiles.yaml`, orchestrator configs, infrastructure
   - ML Platform Engineers: Own MBT plugins and executor backends
4. **Modular Adapter Architecture** - Core is framework-agnostic; all integrations are separate packages
5. **AutoML-First** - Default to AutoML (H2O) for rapid prototyping, but support manual tuning when needed

### Two-Layer Architecture

```
LAYER 1: DECLARATION (Data Scientist)
  ↓ mbt compile
LAYER 2: EXECUTION (Framework + Data Engineering)
```

Data scientists never write DAG code. The compiler generates the execution plan.

## Project Structure

```
mbt/
├── mbt-core/              # Core framework (CLI, compiler, runner)
│   ├── src/mbt/
│   │   ├── cli.py         # Click-based CLI entry point
│   │   ├── core/          # Compiler, runner, manifest, context, DAG
│   │   ├── steps/         # Built-in pipeline steps (load, split, train, etc.)
│   │   ├── contracts/     # Abstract interfaces for adapters
│   │   ├── config/        # YAML schema, loader, profiles
│   │   ├── builtins/      # Default implementations (local storage, env secrets)
│   │   ├── testing/       # Test utilities and fixtures
│   │   └── observability/ # Logging and metrics
│   └── tests/
├── mbt-sklearn/           # Scikit-learn framework adapter
├── mbt-h2o/              # H2O AutoML framework adapter
├── mbt-mlflow/           # MLflow model registry adapter
├── examples/             # Example projects
│   └── telecom-churn/    # Reference implementation
└── BLUEPRINT.md          # Comprehensive architecture documentation (108KB!)
```

## Package Management

- **Package Manager:** `uv` (migrated from pip/poetry)
- **Python Version:** 3.10+ (see `.python-version`)
- **Monorepo:** Multiple packages in one repo, managed with workspace dependencies

### Common Commands

```bash
# Install all packages in development mode
uv sync

# Install specific adapter
uv pip install -e mbt-sklearn/

# Run tests
uv run pytest mbt-core/tests/

# Run MBT CLI
uv run mbt --help
```

## Key Components

### 1. Core Pipeline Steps (`mbt-core/src/mbt/steps/`)

Each step is a Python class inheriting from `BaseStep`:
- `load_data.py` - Load from data sources (CSV, Snowflake, etc.)
- `split_data.py` - Train/test/validation splitting
- `validate_data.py` - Data quality checks
- `normalize.py`, `encode.py` - Feature preprocessing
- `feature_selection.py` - Feature engineering
- `train_model.py` - Model training (delegates to framework adapter)
- `evaluate.py` - Metrics and evaluation
- `predict.py` - Batch scoring
- `log_run.py` - Artifact logging (delegates to registry adapter)
- `load_model.py`, `apply_transforms.py` - Serving pipeline steps

### 2. Contracts (Adapter Interfaces)

Located in `mbt-core/src/mbt/contracts/`:
- `FrameworkAdapter` - ML framework integration (sklearn, H2O, XGBoost)
- `ModelRegistry` - Artifact storage (MLflow, Weights & Biases)
- `DataConnector` - Data source access (Snowflake, BigQuery, Redshift)
- `StorageBackend` - Intermediate artifact storage
- `SecretManager` - Credential management
- `Orchestrator` - DAG generation (Airflow, Prefect, Dagster)

### 3. Compiler (`mbt-core/src/mbt/core/compiler.py`)

Transforms YAML → Manifest:
1. Parse pipeline YAML
2. Resolve profiles and targets
3. Build step graph based on enabled features
4. Validate configurations (call adapter `validate_config()`)
5. Generate `manifest.json` with resolved executors and dependencies

### 4. Runner (`mbt-core/src/mbt/core/runner.py`)

Executes compiled manifest:
1. Load manifest
2. Topologically sort DAG
3. Execute steps in order
4. Pass artifacts via `RunContext`
5. Handle errors and cleanup

## Development Conventions

### Code Style

- **Type Hints:** Use type hints for all public APIs
- **Docstrings:** Google-style docstrings for classes and public methods
- **Naming:**
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`
- **Imports:** Absolute imports preferred; group stdlib, third-party, local

### Configuration Patterns

1. **YAML Schema Validation:** Use Pydantic models in `mbt-core/src/mbt/config/schema.py`
2. **Profile Merging:** Base config + environment-specific overrides
3. **Secrets:** Never in YAML or code; use `SecretManager` contract

### Adapter Development Pattern

```python
from mbt.contracts.framework import FrameworkAdapter

class SklearnFrameworkAdapter(FrameworkAdapter):
    def validate_config(self, config: dict) -> None:
        """Validate config before any execution"""
        pass

    def train(self, X, y, config: dict):
        """Train model and return trained object"""
        pass

    def predict(self, model, X):
        """Generate predictions"""
        pass
```

### Testing Strategy

- **Unit Tests:** Test individual steps and components in isolation
- **Integration Tests:** Test adapter integrations (sklearn, H2O, MLflow)
- **End-to-End Tests:** Full pipeline execution with example projects
- **Fixtures:** Use `mbt.testing.fixtures` for common test data

## Common Workflows

### Adding a New Pipeline Step

1. Create step class in `mbt-core/src/mbt/steps/new_step.py`
2. Inherit from `BaseStep`
3. Implement `execute(context: RunContext) -> dict`
4. Register in compiler's step graph logic
5. Add YAML schema support in `config/schema.py`
6. Write tests in `tests/steps/test_new_step.py`

### Adding a New Framework Adapter

1. Create new package: `mbt-<framework>/`
2. Implement `FrameworkAdapter` interface
3. Add `pyproject.toml` with `mbt-core` dependency
4. Register via entry points or explicit imports
5. Add integration tests
6. Update documentation

### Running Example Pipeline

```bash
cd examples/telecom-churn/

# Compile pipeline
uv run mbt compile churn_simple_v1

# Run locally
uv run mbt run --select churn_simple_v1

# Run with specific profile
uv run mbt run --select churn_simple_v1 --profile dev

# Check results
cat local_artifacts/run_*/metrics.json
```

## Important Files to Read

1. **[BLUEPRINT.md](BLUEPRINT.md)** (108KB) - Complete architecture specification
2. **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Phased development plan
3. **[examples/telecom-churn/pipelines/churn_simple_v1.yaml](examples/telecom-churn/pipelines/churn_simple_v1.yaml)** - Reference pipeline
4. **[mbt-core/src/mbt/core/compiler.py](mbt-core/src/mbt/core/compiler.py)** - Core compilation logic
5. **[mbt-core/src/mbt/config/schema.py](mbt-core/src/mbt/config/schema.py)** - YAML schema definitions

## Key Abstractions

### RunContext
Central data structure passed between steps containing:
- Loaded datasets (train/test/validation)
- Trained models
- Evaluation metrics
- Feature transformers
- Metadata and configurations

### Manifest
JSON file containing:
- Resolved step DAG with dependencies
- Executor configurations (local vs distributed)
- Resource requirements
- Profile-resolved connection strings

### Data Abstraction Layer
Unified interface for data access:
- `DataConnector.read_table()` - Framework-agnostic data loading
- Returns pandas DataFrame or polars DataFrame
- Connector handles SQL generation, authentication, caching

## Known Patterns

### Conditional Step Inclusion
Steps are only added to DAG if enabled:
```yaml
preprocessing:
  normalization:
    enabled: true  # Step included
  feature_selection:
    enabled: false  # Step excluded from DAG
```

### Artifact Passing
Steps communicate via RunContext, not direct file I/O:
```python
# In split_data step
context.set_data("train", X_train, y_train)

# In train_model step
X_train, y_train = context.get_data("train")
```

### Profile Resolution
```yaml
# profiles.yaml
targets:
  dev:
    storage: local
    data_connector: csv
  prod:
    storage: s3
    data_connector: snowflake
    warehouse: COMPUTE_WH
```

## Anti-Patterns to Avoid

1. **Don't put infrastructure logic in core** - Use adapter pattern
2. **Don't hardcode paths** - Use profile-based configuration
3. **Don't skip validation** - Always implement `validate_config()`
4. **Don't break the compilation boundary** - YAML should never contain Python code
5. **Don't couple to specific frameworks** - Core should be framework-agnostic
6. **Don't add unnecessary dependencies to mbt-core** - Keep core minimal

## Git Workflow

- **Main Branch:** `main`
- **Development:** Feature branches, merge to main
- **Commits:** Use conventional commits (feat:, fix:, docs:, refactor:)
- **Status:** Project has recent migration to `uv` package manager (see commit history)

## Current State

Based on `PHASE*_COMPLETE.md` files:
- ✅ Phase 1-4: Core framework, compilation, basic execution implemented
- ✅ Adapters: sklearn, H2O, MLflow basic implementations exist
- 🚧 Active development with working example pipeline
- 📝 Comprehensive documentation in BLUEPRINT.md

## Testing the Framework

```bash
# Run core tests
uv run pytest mbt-core/tests/ -v

# Run integration tests
uv run pytest mbt-sklearn/tests/ -v

# Test with example
cd examples/telecom-churn/
uv run mbt compile churn_simple_v1 && \
uv run mbt run --select churn_simple_v1
```

## Debugging Tips

1. **Compilation errors:** Check YAML schema validation in `config/schema.py`
2. **Runtime errors:** Inspect `RunContext` state between steps
3. **Manifest inspection:** Read generated `manifest.json` to see compiled DAG
4. **Adapter issues:** Verify `validate_config()` implementation
5. **Missing dependencies:** Check `pyproject.toml` and run `uv sync`

## External References

- **Inspiration:** dbt (profiles, CLI), Ludwig (declarative ML), Kedro (catalog), MLflow Recipes
- **AutoML:** H2O.ai for default AutoML backend
- **Orchestrators:** Airflow, Prefect, Dagster (via manifest.json)

---

**Last Updated:** 2026-02-16
**Maintained By:** Project contributors
**Questions?** See BLUEPRINT.md or examples/
