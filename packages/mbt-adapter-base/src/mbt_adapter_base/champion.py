"""The champion record: what a registered version carries, as a model (A-5).

``RegistryAdapter.register`` takes ``metadata: dict[str, str]``, and there were
53 distinct ``mbt.*`` keys across 11 files in 2 packages. Exactly one had a
name - ``promote.OOT_CHECK_TAG`` - and it was written back as a raw literal
anyway, so the constant and the literal sat four lines apart in two files.

The load-bearing champion contract - which artifact, which hooks hash, whether
gates passed, where the baseline lives - was conveyed by key SPELLING. A typo on
the write side was caught only by whichever e2e run happened to read that key
back, and there was no ``RegistryAdapterCompliance``, so a second registry
adapter had nothing to build against.

``ChampionRecord`` owns them. ``pack()`` produces the backend tags, ``unpack()``
reads them back, and the codec for an ``ArtifactRef`` - the same four
``_uri`` / ``_format`` / ``_content_hash`` / ``_size_bytes`` keys, hand-written
three times - exists once as ``pack_artifact`` / ``unpack_artifact``.

**Compatibility is the point of the key names.** They are exactly the strings
the previous code wrote, because champions registered by older mbt must keep
resolving: this is a refactor of who owns the spelling, not a change to it.
"""

from dataclasses import dataclass, field
from typing import Any

from mbt_adapter_base.interchange import ArtifactRef

#: Prefix every mbt-written registry tag carries.
TAG_PREFIX = "mbt."


def _key(name: str) -> str:
    return f"{TAG_PREFIX}{name}"


#: Provenance: what the model was built from.
CONFIG_HASH = _key("config_hash")
INPUT_HASH = _key("input_hash")
MANIFEST_HASH = _key("manifest_hash")
SNAPSHOT_ID = _key("snapshot_id")
GIT_COMMIT = _key("git_commit")
TRACKING_RUN_ID = _key("tracking_run_id")
#: Scoring-time feature-transform parity (ADR-20).
HOOKS_HASH = _key("hooks_hash")
#: Whether the training run's gates passed - the promotion precondition.
GATES_PASSED = _key("gates_passed")
#: The training report, when one was published (ADR-30).
REPORT_URI = _key("report_uri")

#: The after-test verdict (ADR-30). ``promote`` refuses a "false" outright.
OOT_CHECK_PASSED = _key("oot_check.passed")
OOT_CHECK_ANCHOR = _key("oot_check.anchor")
OOT_CHECK_SOURCE = _key("oot_check.source")
OOT_CHECK_RUN_ID = _key("oot_check.run_id")
OOT_CHECK_REPORT_URI = _key("oot_check.report_uri")

#: Prefix for a persisted operating point (R2-5), one per threshold metric.
OPERATING_POINT_PREFIX = _key("operating_point.")

#: The three artifacts a champion points at. Each expands to four keys.
ARTIFACT = _key("artifact")
BASELINE = _key("baseline")
INFERENCE_CONFIG = _key("inference_config")

#: Suffixes of the ArtifactRef codec, in the order they are written.
_REF_SUFFIXES = ("_uri", "_format", "_content_hash", "_size_bytes")


def pack_artifact(prefix: str, ref: ArtifactRef) -> dict[str, str]:
    """An ``ArtifactRef`` as its four tags.

    ``job_result.artifact`` used to be shredded into four strings at one frame
    and reassembled four frames later, with the codec written out three times
    (artifact, baseline, inference config).
    """
    return {
        f"{prefix}_uri": ref.uri,
        f"{prefix}_format": ref.format,
        f"{prefix}_content_hash": ref.content_hash,
        f"{prefix}_size_bytes": str(ref.size_bytes),
    }


def unpack_artifact(prefix: str, tags: dict[str, str]) -> ArtifactRef | None:
    """The ``ArtifactRef`` those four tags describe, or None if absent.

    Returning None rather than raising is the contract: a baseline or inference
    config is optional, and a champion registered by an older mbt legitimately
    carries neither.
    """
    uri = tags.get(f"{prefix}_uri")
    if not uri:
        return None
    return ArtifactRef(
        uri=uri,
        format=tags.get(f"{prefix}_format", "json"),
        content_hash=tags.get(f"{prefix}_content_hash", ""),
        size_bytes=int(tags.get(f"{prefix}_size_bytes", "0") or 0),
    )


@dataclass(frozen=True)
class AfterTestVerdict:
    """The recorded outcome of an after-test check (ADR-30).

    ``passed`` is a THREE-valued string - ``"true"``, ``"false"``, or
    ``"not_gated"`` - because "nothing mature to judge" is a different fact
    from "judged and failed", and ``promote`` treats them differently.
    """

    passed: str
    anchor: str
    source: str
    run_id: str | None = None
    report_uri: str | None = None

    def pack(self) -> dict[str, str]:
        tags = {
            OOT_CHECK_PASSED: self.passed,
            OOT_CHECK_ANCHOR: self.anchor,
            OOT_CHECK_SOURCE: self.source,
        }
        if self.run_id:
            tags[OOT_CHECK_RUN_ID] = self.run_id
        if self.report_uri:
            tags[OOT_CHECK_REPORT_URI] = self.report_uri
        return tags

    @classmethod
    def unpack(cls, tags: dict[str, str]) -> "AfterTestVerdict | None":
        passed = tags.get(OOT_CHECK_PASSED)
        if passed is None:
            return None
        return cls(
            passed=passed,
            anchor=tags.get(OOT_CHECK_ANCHOR, "?"),
            source=tags.get(OOT_CHECK_SOURCE, ""),
            run_id=tags.get(OOT_CHECK_RUN_ID),
            report_uri=tags.get(OOT_CHECK_REPORT_URI),
        )


@dataclass(frozen=True)
class ChampionRecord:
    """Everything mbt records about one registered model version.

    Core builds one and hands it to the registry adapter, which dumps it to
    backend tags. Core stops parsing strings, and the spelling of a key is one
    module's business rather than eleven files'.
    """

    artifact: ArtifactRef
    config_hash: str = ""
    input_hash: str = ""
    manifest_hash: str = ""
    snapshot_id: str = ""
    git_commit: str = ""
    tracking_run_id: str = ""
    hooks_hash: str = ""
    gates_passed: bool = True
    baseline: ArtifactRef | None = None
    inference_config: ArtifactRef | None = None
    report_uri: str | None = None
    after_test: AfterTestVerdict | None = None
    #: metric name -> cutoff, e.g. ``threshold_at_precision_0.35`` (R2-5).
    operating_points: dict[str, str] = field(default_factory=dict)
    #: Anything a caller wants recorded that this model does not name.
    extra: dict[str, str] = field(default_factory=dict)

    def pack(self) -> dict[str, str]:
        """The backend tags for this record."""
        tags: dict[str, str] = {
            CONFIG_HASH: self.config_hash,
            INPUT_HASH: self.input_hash,
            MANIFEST_HASH: self.manifest_hash,
            SNAPSHOT_ID: self.snapshot_id,
            GIT_COMMIT: self.git_commit,
            TRACKING_RUN_ID: self.tracking_run_id,
            GATES_PASSED: "true" if self.gates_passed else "false",
            HOOKS_HASH: self.hooks_hash,
            **pack_artifact(ARTIFACT, self.artifact),
        }
        if self.baseline is not None:
            tags.update(pack_artifact(BASELINE, self.baseline))
        if self.inference_config is not None:
            tags.update(pack_artifact(INFERENCE_CONFIG, self.inference_config))
        if self.report_uri:
            tags[REPORT_URI] = self.report_uri
        if self.after_test is not None:
            tags.update(self.after_test.pack())
        for name, value in self.operating_points.items():
            tags[f"{OPERATING_POINT_PREFIX}{name}"] = value
        tags.update(self.extra)
        return tags

    @classmethod
    def unpack(cls, tags: dict[str, str]) -> "ChampionRecord":
        """Read a record back from backend tags.

        The artifact is the one required part; a version whose tags carry no
        artifact uri is not a champion mbt registered.
        """
        artifact = unpack_artifact(ARTIFACT, tags)
        if artifact is None:
            raise ValueError(f"no {ARTIFACT}_uri tag: not a champion mbt registered")
        return cls(
            artifact=artifact,
            config_hash=tags.get(CONFIG_HASH, ""),
            input_hash=tags.get(INPUT_HASH, ""),
            manifest_hash=tags.get(MANIFEST_HASH, ""),
            snapshot_id=tags.get(SNAPSHOT_ID, ""),
            git_commit=tags.get(GIT_COMMIT, ""),
            tracking_run_id=tags.get(TRACKING_RUN_ID, ""),
            hooks_hash=tags.get(HOOKS_HASH, ""),
            gates_passed=tags.get(GATES_PASSED) == "true",
            baseline=unpack_artifact(BASELINE, tags),
            inference_config=unpack_artifact(INFERENCE_CONFIG, tags),
            report_uri=tags.get(REPORT_URI),
            after_test=AfterTestVerdict.unpack(tags),
            operating_points={
                key.removeprefix(OPERATING_POINT_PREFIX): value
                for key, value in tags.items()
                if key.startswith(OPERATING_POINT_PREFIX)
            },
        )


def has_inference_config(tags: dict[str, str]) -> bool:
    """Whether a version carries an inference config (ADR-28).

    ``read_inference_config``'s docstring used to say "callers check the tag
    exists first", an obligation held in prose and honoured by one of its
    callers (A-5). This is that check, so the other callers can honour it too.
    """
    return bool(tags.get(f"{INFERENCE_CONFIG}_uri"))


def operating_points(metrics: dict[str, float] | Any) -> dict[str, str]:
    """The deployable cutoffs among a metric result set (R2-5).

    A ``threshold_at_*`` metric is a SCORE CUTOFF, not a quality measure, so it
    is persisted for a scoring pipeline to default its decision rule from rather
    than compared against a gate.
    """
    return {
        name: str(value)
        for name, value in dict(metrics).items()
        if name.startswith(("threshold_at_precision_", "threshold_at_recall_"))
    }


__all__ = [
    "ARTIFACT",
    "BASELINE",
    "CONFIG_HASH",
    "GATES_PASSED",
    "GIT_COMMIT",
    "HOOKS_HASH",
    "INFERENCE_CONFIG",
    "INPUT_HASH",
    "MANIFEST_HASH",
    "OOT_CHECK_ANCHOR",
    "OOT_CHECK_PASSED",
    "OOT_CHECK_REPORT_URI",
    "OOT_CHECK_RUN_ID",
    "OOT_CHECK_SOURCE",
    "OPERATING_POINT_PREFIX",
    "REPORT_URI",
    "SNAPSHOT_ID",
    "TAG_PREFIX",
    "TRACKING_RUN_ID",
    "AfterTestVerdict",
    "ChampionRecord",
    "has_inference_config",
    "operating_points",
    "pack_artifact",
    "unpack_artifact",
]
