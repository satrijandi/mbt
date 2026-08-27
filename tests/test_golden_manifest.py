"""Golden-file compile test (TSD §21, S2-09).

tests/fixtures/churn_demo compiles to a checked-in manifest with volatile metadata
normalized; any diff is a reviewable change. Regenerate deliberately with:

    UPDATE_GOLDEN=1 uv run pytest tests/test_golden_manifest.py
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from e2e_utils import CHURN_DEMO, DEMO_ANCHOR

from mbt.compile.compiler import CompileOptions, compile_project
from mbt.config.profiles import load_profiles
from mbt.parsing import parse_project

GOLDEN = Path(__file__).parent / "golden" / "churn_demo_manifest.json"


def normalize(payload: dict[str, Any]) -> dict[str, Any]:
    """Blank machine/checkout-dependent fields; keep everything semantic."""
    metadata = payload["metadata"]
    metadata["env_digest"] = "<env>"
    metadata["env_freeze_digest"] = "<env>"
    metadata["git"] = {"branch": None, "commit": None, "dirty": False}
    for version in payload.get("adapter_versions", {}).values():
        version["version"] = "<version>"
    return payload


def compile_churn_demo() -> dict[str, Any]:
    parsed = parse_project(CHURN_DEMO)
    profiles = load_profiles(
        "churn_demo", CHURN_DEMO, target_override="ci", project_vars=parsed.project.vars
    )
    manifest = compile_project(
        parsed,
        profiles,
        options=CompileOptions(
            anchor=datetime.fromisoformat(DEMO_ANCHOR.replace("Z", "+00:00")),
            deep_snapshot=True,  # content-hashed: stable across checkouts (ADR-11)
        ),
    )
    return normalize(json.loads(manifest.to_json()))


def test_churn_demo_matches_golden_manifest() -> None:
    current = compile_churn_demo()
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    assert GOLDEN.is_file(), "golden manifest missing; run once with UPDATE_GOLDEN=1"
    golden = json.loads(GOLDEN.read_text())
    assert current == golden, (
        "compiled manifest differs from the golden file; if the change is "
        "intentional, regenerate with UPDATE_GOLDEN=1 and review the diff"
    )


def test_compile_is_byte_deterministic() -> None:
    assert compile_churn_demo() == compile_churn_demo()
