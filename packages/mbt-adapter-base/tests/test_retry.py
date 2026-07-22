"""retry_with_jitter: the shared transient-error backoff for data-plane seams
(R2-2). Each seam supplies its own transient/deterministic predicate; this only
pins the loop and backoff contract."""

import pytest

from mbt_adapter_base import retry_with_jitter


class _Transient(Exception):
    pass


class _Fatal(Exception):
    pass


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, _Transient)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    # keep the jittered backoff from actually sleeping in tests
    monkeypatch.setattr("mbt_adapter_base.retry.time.sleep", lambda _s: None)


def test_returns_immediately_on_success() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        return "ok"

    assert retry_with_jitter(fn, is_transient=_is_transient) == "ok"
    assert calls["n"] == 1  # called exactly once, no retry


def test_retries_a_transient_error_then_succeeds() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Transient("blip")
        return "ok"

    assert retry_with_jitter(fn, is_transient=_is_transient, attempts=6) == "ok"
    assert calls["n"] == 3  # two transient failures, third try succeeds


def test_a_non_transient_error_is_not_retried() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise _Fatal("bad sql")

    with pytest.raises(_Fatal):
        retry_with_jitter(fn, is_transient=_is_transient)
    assert calls["n"] == 1  # raised on the first attempt, never retried


def test_exhausting_the_attempts_propagates_the_last_error() -> None:
    calls = {"n": 0}

    def fn() -> str:
        calls["n"] += 1
        raise _Transient("still down")

    with pytest.raises(_Transient):
        retry_with_jitter(fn, is_transient=_is_transient, attempts=3)
    assert calls["n"] == 3  # attempts-1 retries in the loop, then the final bare try
