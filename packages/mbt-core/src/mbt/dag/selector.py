"""Node selector grammar and evaluation (TSD §9.2, FR-DAG-02/03/04).

Grammar (dbt semantics: comma is intersection, space is union)::

    spec         := union
    union        := intersection { " " intersection }
    intersection := atom { "," atom }
    atom         := [ [digits] "+" ] body [ "+" [digits] ]
    body         := method ":" value | name_glob
    method       := "tag" | "state" | "resource_type"
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Protocol

import networkx as nx

from mbt.exceptions import MbtError


class SelectorError(MbtError):
    """Invalid selector syntax or missing prerequisites (e.g. --state)."""


class StateIndex(Protocol):
    """Answers state:new / state:modified against a reference manifest."""

    def is_new(self, unique_id: str) -> bool: ...

    def is_modified(self, unique_id: str) -> bool: ...


@dataclass(frozen=True)
class SelectableNode:
    """The minimal node view selectors evaluate against."""

    unique_id: str
    name: str
    resource_type: str
    tags: tuple[str, ...] = ()


_ATOM_RE = re.compile(r"^(?:(?P<updepth>\d*)\+)?(?P<body>[^+]+?)(?:\+(?P<downdepth>\d*))?$")
_METHODS = ("tag", "state", "resource_type")
_STATE_VALUES = ("new", "modified")


@dataclass(frozen=True)
class Atom:
    body_method: str | None  # None for name globs
    body_value: str
    up: bool = False  # leading '+' present
    up_depth: int | None = None  # None = unlimited
    down: bool = False  # trailing '+' present
    down_depth: int | None = None


@dataclass(frozen=True)
class Selector:
    """Parsed selector: union of intersections of atoms."""

    union: tuple[tuple[Atom, ...], ...]
    text: str


def parse_selector(text: str) -> Selector:
    """Parse a selector string (spaces = union, commas = intersection)."""
    text = text.strip()
    if not text:
        raise SelectorError("empty selector")
    union: list[tuple[Atom, ...]] = []
    for intersection_text in text.split():
        atoms: list[Atom] = []
        for atom_text in intersection_text.split(","):
            atoms.append(_parse_atom(atom_text, text))
        union.append(tuple(atoms))
    return Selector(union=tuple(union), text=text)


def _parse_atom(atom_text: str, full: str) -> Atom:
    if not atom_text:
        raise SelectorError(
            f"empty selector atom in {full!r}",
            hint="did you write a stray comma?",
        )
    match = _ATOM_RE.match(atom_text)
    if match is None:
        raise SelectorError(f"invalid selector atom {atom_text!r} in {full!r}")
    up_raw = match.group("updepth")  # None: no leading '+'; '': unlimited; digits: depth
    down_raw = match.group("downdepth")
    body = match.group("body")

    up = up_raw is not None
    up_depth = int(up_raw) if up_raw else None
    down = down_raw is not None
    down_depth = int(down_raw) if down_raw else None

    if ":" in body:
        method, _, value = body.partition(":")
        if method not in _METHODS:
            raise SelectorError(
                f"unknown selector method {method!r} in {atom_text!r}",
                hint=f"methods: {', '.join(_METHODS)}",
            )
        if not value:
            raise SelectorError(f"selector method {method!r} needs a value in {atom_text!r}")
        if method == "state" and value not in _STATE_VALUES:
            raise SelectorError(
                f"unknown state selector value {value!r}",
                hint=f"state values: {', '.join(_STATE_VALUES)}",
            )
        return Atom(
            body_method=method,
            body_value=value,
            up=up,
            up_depth=up_depth,
            down=down,
            down_depth=down_depth,
        )
    return Atom(
        body_method=None,
        body_value=body,
        up=up,
        up_depth=up_depth,
        down=down,
        down_depth=down_depth,
    )


def _match_atom_base(
    atom: Atom,
    nodes: dict[str, SelectableNode],
    state: StateIndex | None,
) -> set[str]:
    if atom.body_method is None:
        pattern = atom.body_value
        return {
            uid
            for uid, node in nodes.items()
            if fnmatchcase(node.name, pattern) or fnmatchcase(uid, pattern)
        }
    if atom.body_method == "tag":
        return {uid for uid, node in nodes.items() if atom.body_value in node.tags}
    if atom.body_method == "resource_type":
        return {uid for uid, node in nodes.items() if node.resource_type == atom.body_value}
    # state:
    if state is None:
        raise SelectorError(
            f"selector 'state:{atom.body_value}' requires --state <path-or-URI> "
            "pointing at a reference manifest",
            hint="e.g. --state s3://bucket/mbt/proj/prod/manifests/latest.json",
        )
    if atom.body_value == "new":
        return {uid for uid in nodes if state.is_new(uid)}
    return {uid for uid in nodes if state.is_modified(uid)}


def _expand_graph(base: set[str], graph: "nx.DiGraph", atom: Atom) -> set[str]:
    selected = set(base)
    if atom.up:
        for uid in base:
            if uid not in graph:
                continue
            if atom.up_depth is None:
                selected |= nx.ancestors(graph, uid)
            else:
                lengths = nx.single_source_shortest_path_length(
                    graph.reverse(copy=False), uid, cutoff=atom.up_depth
                )
                selected |= set(lengths)
    if atom.down:
        for uid in base:
            if uid not in graph:
                continue
            if atom.down_depth is None:
                selected |= nx.descendants(graph, uid)
            else:
                lengths = nx.single_source_shortest_path_length(graph, uid, cutoff=atom.down_depth)
                selected |= set(lengths)
    return selected


def evaluate_selector(
    selector: Selector | str,
    graph: "nx.DiGraph",
    nodes: dict[str, SelectableNode],
    state: StateIndex | None = None,
) -> set[str]:
    """Evaluate a selector against nodes; returns matching unique_ids.

    Graph expansion (+ operators) may pull in unique_ids outside ``nodes``
    (e.g. sources); callers typically intersect with executable nodes.
    """
    if isinstance(selector, str):
        selector = parse_selector(selector)
    result: set[str] = set()
    for intersection in selector.union:
        atom_sets: list[set[str]] = []
        for atom in intersection:
            base = _match_atom_base(atom, nodes, state)
            atom_sets.append(_expand_graph(base, graph, atom))
        common = set.intersection(*atom_sets) if atom_sets else set()
        result |= common
    return result


def _check_atoms_resolve(selector: Selector, nodes: dict[str, SelectableNode], flag: str) -> None:
    """Every name, tag and resource_type atom must name something in the project
    (FEEDBACK v6 A-5).

    ``mbt build --select churn_clasifier`` used to select 0 nodes and exit 0,
    and the scaffold's scheduled scoring runs ``--select tag:daily`` then pings
    a dead-man's switch on success: renaming the tag stopped scoring and kept
    the heartbeat alive. An atom that resolves to no resource is a typo or a
    rename, never an intent, so it is an error with the same did-you-mean the
    parser gives.

    Two empties stay legitimate, because the project gives them meaning:
    ``state:`` atoms (nothing modified is the common, correct answer) and an
    intersection of atoms that each match something but share nothing.
    """
    from mbt.utils import did_you_mean

    for intersection in selector.union:
        for atom in intersection:
            if atom.body_method == "state" or _match_atom_base(atom, nodes, None):
                continue
            label = atom.body_value
            if atom.body_method is None:
                kind, candidates = "resource", sorted({n.name for n in nodes.values()})
            elif atom.body_method == "tag":
                kind = "tag"
                candidates = sorted({t for n in nodes.values() for t in n.tags})
                label = f"tag:{label}"
            else:
                kind = "resource_type"
                candidates = sorted({n.resource_type for n in nodes.values()})
                label = f"resource_type:{label}"
            guess = did_you_mean(atom.body_value, candidates)
            hint = f"did you mean {guess!r}? " if guess else ""
            shown = ", ".join(candidates[:20]) + (" ..." if len(candidates) > 20 else "")
            raise SelectorError(
                f"{flag} {label!r} matches no {kind} in this project",
                hint=f"{hint}known {kind}s: {shown or '(none)'}",
            )


def select_nodes(
    graph: "nx.DiGraph",
    nodes: dict[str, SelectableNode],
    select: Iterable[str] | None,
    exclude: Iterable[str] | None = None,
    state: StateIndex | None = None,
) -> set[str]:
    """Full --select/--exclude evaluation (FR-DAG-04).

    An atom naming nothing in the project raises, on either flag: a mistyped
    ``--exclude`` would otherwise run the very thing it was written to skip.
    """
    select_text = " ".join(select) if select else ""
    if select_text.strip():
        parsed = parse_selector(select_text)
        _check_atoms_resolve(parsed, nodes, "--select")
        selected = evaluate_selector(parsed, graph, nodes, state)
    else:
        selected = set(nodes)
    exclude_text = " ".join(exclude) if exclude else ""
    if exclude_text.strip():
        parsed = parse_selector(exclude_text)
        _check_atoms_resolve(parsed, nodes, "--exclude")
        selected -= evaluate_selector(parsed, graph, nodes, state)
    return selected & set(nodes)
