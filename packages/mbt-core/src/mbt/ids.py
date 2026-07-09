"""Stable unique_id construction (TSD §4).

``<resource_type>.<project_name>.<name>`` for datasets/models/scoring/
exposures; sources add their group: ``source.<project>.<group>.<table>``.
"""

RESOURCE_TYPES = ("source", "dataset", "model", "metric", "exposure", "scoring")


def unique_id(resource_type: str, project: str, name: str) -> str:
    if resource_type not in RESOURCE_TYPES:
        raise ValueError(f"unknown resource type: {resource_type!r}")
    return f"{resource_type}.{project}.{name}"


def source_unique_id(project: str, group: str, table: str) -> str:
    return f"source.{project}.{group}.{table}"


def resource_type_of(uid: str) -> str:
    return uid.split(".", 1)[0]


def name_of(uid: str) -> str:
    """The resource name part of a unique_id (last segment)."""
    return uid.rsplit(".", 1)[-1]
