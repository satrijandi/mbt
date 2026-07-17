"""Property-based invariants for canonicalization and identity hashing.

canonical_json / config_hash / input_hash drive every gate, golden, and
state:modified decision (ADR-4/6/7). Their byte-reproducibility had been
guarded only by a handful of example asserts plus a manual audit. These
Hypothesis properties pin the invariants the audit checked by hand:
order-independence, idempotence, and insensitivity to cosmetic fields.
"""

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from mbt.compile.hashing import HASH_EXCLUDED_FIELDS, config_hash, input_hash
from mbt.utils import canonical_json

# JSON-serializable values (no NaN/inf: canonical_json intentionally has no
# allow_nan escape, and NaN != NaN would break the round-trip properties).
_scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)
_json = st.recursive(
    _scalars,
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=12,
)
_json_dicts = st.dictionaries(st.text(), _json)

# config_hash / snapshot_id values are always sha256-prefixed tokens in practice
# (never contain the '|' that input_hash joins on), so injectivity is meaningful
# only over that realistic alphabet.
_tokens = st.text(alphabet="0123456789abcdef:sha", min_size=1, max_size=24)


@given(pairs=st.lists(st.tuples(st.text(), _json), unique_by=lambda kv: kv[0]))
def test_canonical_json_is_key_order_independent(pairs: list[tuple[str, Any]]) -> None:
    forward = dict(pairs)
    backward = dict(reversed(pairs))
    assert canonical_json(forward) == canonical_json(backward)


@given(value=_json)
def test_canonical_json_is_an_idempotent_fixpoint(value: Any) -> None:
    import json

    once = canonical_json(value)
    twice = canonical_json(json.loads(once))
    assert once == twice


@given(pairs=st.lists(st.tuples(st.text(), _json), unique_by=lambda kv: kv[0]))
def test_config_hash_is_key_order_independent(pairs: list[tuple[str, Any]]) -> None:
    assert config_hash(dict(pairs)) == config_hash(dict(reversed(pairs)))


@given(cfg=_json_dicts, description=st.text(), owner=st.text(), tags=st.lists(st.text()))
def test_config_hash_ignores_cosmetic_fields(
    cfg: dict[str, Any], description: str, owner: str, tags: list[str]
) -> None:
    base = {k: v for k, v in cfg.items() if k not in HASH_EXCLUDED_FIELDS}
    cosmetic = {**base, "description": description, "owner": owner, "tags": tags}
    assert config_hash(base) == config_hash(cosmetic)


@given(ch=_tokens, snap=_tokens, upstream=st.lists(_tokens))
def test_input_hash_is_upstream_order_independent(ch: str, snap: str, upstream: list[str]) -> None:
    assert input_hash(ch, snap, list(upstream)) == input_hash(ch, snap, list(reversed(upstream)))


@given(ch1=_tokens, ch2=_tokens, snap=_tokens, upstream=st.lists(_tokens))
def test_input_hash_tracks_config_hash_changes(
    ch1: str, ch2: str, snap: str, upstream: list[str]
) -> None:
    if ch1 == ch2:
        return
    assert input_hash(ch1, snap, upstream) != input_hash(ch2, snap, upstream)
