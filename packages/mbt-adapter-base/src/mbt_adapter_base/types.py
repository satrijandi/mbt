"""Common enums and sentinels shared by mbt-core and all adapters (TSD §5.1)."""

from enum import StrEnum

#: Version of the adapter contract defined by this package (TSD §12, §19).
#: Plugins declare the contract version they were built against; core accepts
#: the same major with a minor less than or equal to its own.
#: 1.2 (ADR-30) adds the optional ``reporting`` role; a 1.1 plugin still loads.
CONTRACT_VERSION = "1.2"

#: What ``{{ auto }}`` renders to in a spec. Hyperparameter values equal to
#: this sentinel skip static validation and are resolved by the adapter's
#: ``resolve_auto`` from the dataset profile at run time (FR-RES-10).
AUTO = "__mbt_auto__"

#: The after-test split (ADR-30): rows from the end of the test window up to
#: the anchor. Scored for stability and per-period performance, never trained
#: on, never tuned against, and - unlike every other split - allowed to be
#: empty and to carry labels that have not matured yet.
OUT_OF_TIME_SPLIT = "out_of_time"


class TaskType(StrEnum):
    """ML task kinds. The ``task`` field of a model selects its task schema."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"  # stretch
    REGRESSION = "regression"  # stretch
    RANKING = "ranking"  # v1
    SURVIVAL = "survival"  # v1


class SplitStrategy(StrEnum):
    """Dataset split strategies. Temporal is the default (FR-RES-09)."""

    TEMPORAL = "temporal"
    RANDOM = "random"  # must be opted into explicitly


class Materialization(StrEnum):
    """How a trained model materializes. v0 ships ``model_artifact`` only."""

    MODEL_ARTIFACT = "model_artifact"
    ENSEMBLE = "ensemble"  # v1
    CALIBRATED = "calibrated"  # v1
    ONNX = "onnx"  # v1 (export path proven in v0)


class Stage(StrEnum):
    """Canonical registry stage tokens; registry adapters map them (TSD §13.3)."""

    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
