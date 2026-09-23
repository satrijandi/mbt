"""The champion record owns 53 keys' worth of spelling (A-5)."""

import pytest

from mbt_adapter_base.base import ArrowTrainingAdapter, ShapArrowTrainingAdapter
from mbt_adapter_base.champion import (
    ARTIFACT,
    OOT_CHECK_PASSED,
    OOT_CHECK_REPORT_URI,
    OOT_CHECK_RUN_ID,
    AfterTestVerdict,
    ChampionRecord,
    has_inference_config,
    operating_points,
    unpack_artifact,
)
from mbt_adapter_base.events import AdapterMessage
from mbt_adapter_base.interchange import ArtifactRef


def _ref(uri: str = "file:///tmp/m.json") -> ArtifactRef:
    return ArtifactRef(uri=uri, format="native", content_hash="sha256:aa", size_bytes=3)


def test_a_verdict_carries_its_optional_fields_when_set() -> None:
    verdict = AfterTestVerdict(
        passed="true", anchor="a", source="check", run_id="r1", report_uri="file:///r"
    )
    tags = verdict.pack()
    assert tags[OOT_CHECK_RUN_ID] == "r1"
    assert tags[OOT_CHECK_REPORT_URI] == "file:///r"
    assert AfterTestVerdict.unpack(tags) == verdict


def test_a_verdict_omits_optional_fields_it_does_not_have() -> None:
    tags = AfterTestVerdict(passed="not_gated", anchor="a", source="build").pack()
    assert OOT_CHECK_RUN_ID not in tags and OOT_CHECK_REPORT_URI not in tags
    assert AfterTestVerdict.unpack(tags).passed == "not_gated"


def test_no_verdict_tag_means_no_verdict() -> None:
    assert AfterTestVerdict.unpack({}) is None
    assert AfterTestVerdict.unpack({OOT_CHECK_PASSED: "false"}).anchor == "?"


def test_a_record_round_trips_through_its_tags() -> None:
    record = ChampionRecord(
        artifact=_ref(),
        gates_passed=False,
        baseline=_ref("file:///tmp/b.json"),
        report_uri="file:///tmp/r",
        after_test=AfterTestVerdict(passed="true", anchor="a", source="build"),
        operating_points={"threshold_at_precision_0.5": "0.61"},
    )
    assert ChampionRecord.unpack(record.pack()) == record


def test_tags_with_no_artifact_are_not_a_champion_mbt_registered() -> None:
    with pytest.raises(ValueError, match="not a champion mbt registered"):
        ChampionRecord.unpack({"mbt.config_hash": "x"})


def test_an_optional_artifact_is_absent_rather_than_empty() -> None:
    """A champion registered by an older mbt legitimately carries no baseline."""
    assert unpack_artifact("mbt.baseline", {}) is None
    assert unpack_artifact(ARTIFACT, {f"{ARTIFACT}_uri": "file:///m"}).size_bytes == 0


def test_extra_tags_ride_along() -> None:
    record = ChampionRecord(artifact=_ref(), extra={"mbt.custom": "1"})
    assert record.pack()["mbt.custom"] == "1"


def test_has_inference_config_is_the_prose_obligation_as_a_function() -> None:
    assert not has_inference_config({})
    assert not has_inference_config({"mbt.inference_config_uri": ""})
    assert has_inference_config({"mbt.inference_config_uri": "file:///c.json"})


def test_only_threshold_metrics_are_deployable_cutoffs() -> None:
    picked = operating_points(
        {"roc_auc": 0.8, "threshold_at_precision_0.5": 0.61, "threshold_at_recall_0.2": 0.3}
    )
    assert set(picked) == {"threshold_at_precision_0.5", "threshold_at_recall_0.2"}
    assert picked["threshold_at_precision_0.5"] == "0.61"


def test_an_arrow_adapter_must_implement_scores() -> None:
    """``_scores`` is the one abstract hook everything else derives from (A-4)."""
    with pytest.raises(NotImplementedError):
        ArrowTrainingAdapter()._scores(object(), None)


def test_a_shap_adapter_must_implement_shap_values() -> None:
    with pytest.raises(NotImplementedError):
        ShapArrowTrainingAdapter()._shap_values(object(), None)


def test_an_adapter_message_without_a_name_is_just_its_text() -> None:
    assert AdapterMessage(message="hello").human() == "hello"
    assert AdapterMessage(adapter="h2o", message="hello").human() == "h2o: hello"


def test_the_compliance_bases_require_their_one_hook() -> None:
    """Each suite has exactly one thing a subclass must supply."""
    from pathlib import Path

    from mbt_adapter_base.compliance import DataAdapterCompliance, RegistryAdapterCompliance
    from mbt_adapter_base.compliance.suite import PredictionStoreCompliance

    with pytest.raises(NotImplementedError):
        DataAdapterCompliance().make_adapter(Path("."), None)
    with pytest.raises(NotImplementedError):
        RegistryAdapterCompliance().make_registry(Path("."))
    with pytest.raises(NotImplementedError):
        PredictionStoreCompliance().make_store(Path("."))
