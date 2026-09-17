"""Model and dataset configs as flat tracking parameters (ADR-30).

A tracker's comparison view shows parameters, not documents, so a run whose
config lives only in a JSON artifact cannot be filtered by its test window or
its feature treatment. This turns the resolved configs into dotted keys -
``model.evaluation.protocol.split``, ``dataset.windows.test.start`` - beside
the bare hyperparameter keys runs have always carried.

Keys are only ever built from spec field names, which every tracker accepts.
A mapping keyed by anything else - column names under
``features.transforms``, for instance - is encoded as one JSON value, because
a column name is not always a valid parameter key. Lists are JSON too.
Length limits are the tracker's to enforce; the full documents are logged
beside the parameters so a truncated value always has a source.
"""

import re
from collections.abc import Mapping
from typing import Any

from mbt.secrets import redact
from mbt.utils import canonical_json

#: Keys a tracker can take verbatim: letters, digits, and ``_.-/ :``.
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-/ :]*$")

#: Model spec mappings keyed by column names, not by spec fields.
_USER_KEYED = frozenset(
    {
        "features.categorical",
        "features.transforms",
        "features.monotonic",
    }
)


def _value(value: Any) -> str:
    """Strings verbatim; everything else as canonical JSON (``true``, ``null``, ``[...]``)."""
    text = value if isinstance(value, str) else canonical_json(value)
    return redact(text)


def flatten(
    prefix: str, config: Mapping[str, Any], *, user_keyed: frozenset[str] = _USER_KEYED
) -> dict[str, str]:
    """Dotted parameters for one config mapping, in a stable order."""
    out: dict[str, str] = {}

    def walk(path: str, value: Any) -> None:
        if isinstance(value, Mapping):
            relative = path.split(".", 1)[1] if "." in path else ""
            keys = [str(k) for k in value]
            if relative in user_keyed or not all(_SAFE_KEY.match(k) for k in keys):
                out[path] = _value(dict(value))
                return
            if not value:
                out[path] = "{}"
                return
            for key in sorted(value, key=str):
                walk(f"{path}.{key}", value[key])
            return
        out[path] = _value(value)

    for key in sorted(config, key=str):
        walk(f"{prefix}.{key}", config[key])
    return out


def model_params(resolved_config: Mapping[str, Any], feature_columns: list[str]) -> dict[str, str]:
    """The trained model's spec - AUTO resolved and tuning applied - plus the
    feature columns it was actually fit on, which ``include: ["*"]`` hides."""
    params = flatten("model", resolved_config)
    params["model.resolved.n_features"] = str(len(feature_columns))
    params["model.resolved.feature_columns"] = _value(list(feature_columns))
    return params


def dataset_params(
    config: Mapping[str, Any],
    *,
    windows: Mapping[str, Any],
    anchor: str,
    sample_fraction: float | None,
    row_counts: Mapping[str, int],
) -> dict[str, str]:
    """The dataset spec plus the concrete rows it produced for this run."""
    params = flatten("dataset", config)
    for split, bounds in sorted(windows.items()):
        params[f"dataset.windows.{split}.start"] = str(bounds[0])
        params[f"dataset.windows.{split}.end"] = str(bounds[1])
    params["dataset.anchor"] = anchor
    if sample_fraction is not None:
        params["dataset.sample_fraction"] = _value(sample_fraction)
    for split, count in sorted(row_counts.items()):
        params[f"dataset.rows.{split}"] = str(count)
    return params
