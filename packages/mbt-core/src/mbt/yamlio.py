"""The one YAML reader every mbt config file goes through (FEEDBACK v6 A-4).

PyYAML's ``safe_load`` keeps the LAST of two equal keys in a mapping, silently.
mbt's premise is that the reviewed YAML is the model, and ``promotions.yml`` is
the reviewed control over production, so a duplicate key makes the line a
reviewer reads differ from the value that runs:

    promotions:
      - model: churn_classifier
        version: "3"            # the version the reviewer approved
        to: production
        version: "5"            # the version that ships

Long specs and merge resolutions produce duplicates by accident; a hostile PR
can produce one on purpose. Here a repeated key is a ``yaml.YAMLError`` naming
the file, both lines and the key, so every caller's existing YAML-error path
reports it without knowing it exists.

Merge keys (``<<: *defaults``) keep their YAML meaning: an explicit key that
overrides a merged one is the point of a merge, not a duplicate.
``tests/test_yaml_duplicate_keys.py`` fails the suite if a ``yaml.safe_load``
appears anywhere else in the packages, so a new loader cannot quietly reopen this.
"""

from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node

_MERGE_TAG = "tag:yaml.org,2002:merge"


class DuplicateKeyError(ConstructorError):
    """A mapping repeats a key. A ``yaml.YAMLError``, so callers need no new except."""


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: Node, deep: bool = False) -> dict[Any, Any]:
        if isinstance(node, MappingNode):
            # Before the base class flattens merge keys into node.value, so only
            # keys written in THIS mapping are compared with each other.
            first_seen: dict[Any, Node] = {}
            for key_node, _ in node.value:
                if key_node.tag == _MERGE_TAG:
                    continue
                key = self.construct_object(key_node, deep=True)
                try:
                    earlier = first_seen.get(key)
                except TypeError:
                    continue  # unhashable; the base class raises its own error for it
                if earlier is not None:
                    raise DuplicateKeyError(
                        f"while constructing a mapping (key {key!r} first defined here)",
                        earlier.start_mark,
                        f"found duplicate key {key!r}; YAML would silently keep only "
                        "this last value, so mbt refuses the file",
                        key_node.start_mark,
                    )
                first_seen[key] = key_node
        return super().construct_mapping(node, deep=deep)  # type: ignore[arg-type]


def safe_load(
    text: str, *, source: str | Path | None = None, allow_duplicate_keys: bool = False
) -> Any:
    """``yaml.safe_load`` that refuses a repeated mapping key.

    ``source`` names the file in error marks (``in "models/m.yml", line 5``);
    without it they read ``<unicode string>``. ``allow_duplicate_keys`` is for
    diagnostics only - reading the names out of a file already rejected, so its
    dependents are not reported as dangling too - never for loading config.
    """
    loader = yaml.SafeLoader(text) if allow_duplicate_keys else _UniqueKeyLoader(text)
    if source is not None:
        loader.name = str(source)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


__all__ = ["DuplicateKeyError", "safe_load"]
