"""The MLflow tracker's declared capabilities agree with its methods (FEEDBACK v7 C-1)."""

from pathlib import Path
from typing import Any

from mbt_mlflow.adapter import MlflowTracking

from mbt_adapter_base.capabilities import Capability, supports
from mbt_adapter_base.compliance import TrackingAdapterCompliance


class TestMlflowTrackingCompliance(TrackingAdapterCompliance):
    def make_tracking(self, root: Path) -> Any:
        return MlflowTracking({"uri": f"sqlite:///{root / 'mlflow.db'}"})


def test_mlflow_tracking_declares_all_four_optional_capabilities(tmp_path: Path) -> None:
    tracking = MlflowTracking({"uri": f"sqlite:///{tmp_path / 'mlflow.db'}"})
    for capability in (
        Capability.PREPARE,
        Capability.LOG_TRIAL,
        Capability.LOG_DOCUMENT,
        Capability.LOG_DIRECTORY,
    ):
        assert supports(tracking, capability), capability
