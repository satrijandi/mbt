"""The fake tracker and the in-process compute adapter declare what they implement (C-1)."""

from pathlib import Path
from typing import Any

from mbt_testing import FakeTrackingAdapter
from mbt_testing.adapters import InlineComputeAdapter

from mbt_adapter_base.compliance import ComputeAdapterCompliance, TrackingAdapterCompliance


class TestFakeTrackingCompliance(TrackingAdapterCompliance):
    def make_tracking(self, root: Path) -> Any:
        return FakeTrackingAdapter({"root": str(root / "tracking")})


class TestInlineComputeCompliance(ComputeAdapterCompliance):
    def make_compute(self, root: Path) -> Any:
        return InlineComputeAdapter({})
