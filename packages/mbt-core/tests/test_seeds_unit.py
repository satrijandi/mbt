"""The seed ladder is a registry, so its invariants are testable (B-3).

Before ``execute/seeds.py`` the ladder was inline arithmetic documented in
prose, and the only test proved that *two* rungs differed. Nothing proved the
set was collision-free - which is how ``spec.seed + 3`` came to be taken twice,
by the champion gate's bootstrap and by the tuning objective's (v5 live defect
2). With the ladder as an enum that becomes one test over the whole set.
"""

import inspect

import pytest

from mbt.execute import seeds
from mbt.execute.seeds import SeedRung, seed_for


def test_every_rung_has_a_distinct_offset() -> None:
    """The property that a second ``+3`` violates."""
    offsets = [int(rung) for rung in SeedRung]
    assert len(offsets) == len(set(offsets)), (
        f"two seeded stages share an offset, so they draw the same random stream: {sorted(offsets)}"
    )


def test_rungs_are_contiguous_from_zero() -> None:
    """A new stage takes the NEXT free rung, so a gap means one was renumbered
    or removed - either of which silently changes results for pinned seeds."""
    assert sorted(int(rung) for rung in SeedRung) == list(range(len(SeedRung)))


def test_every_rung_has_a_named_accessor() -> None:
    """Call sites use the names, so a rung with no accessor is unreachable and
    the next author writes the arithmetic inline again."""
    accessors = {
        name
        for name, value in vars(seeds).items()
        if name.endswith("_seed") and inspect.isfunction(value)
    }
    expected = {f"{rung.name.lower()}_seed" for rung in SeedRung}
    # CHAMPION_BOOTSTRAP -> champion_bootstrap_seed, and so on.
    assert expected == accessors


@pytest.mark.parametrize("rung", list(SeedRung))
def test_accessor_agrees_with_its_rung(rung: SeedRung) -> None:
    accessor = getattr(seeds, f"{rung.name.lower()}_seed")
    assert accessor(1000) == seed_for(1000, rung) == 1000 + int(rung)


def test_train_rung_is_the_declared_seed() -> None:
    """``spec.seed`` IS the fit's seed; every other stage offsets from it."""
    assert seeds.train_seed(42) == 42


def test_tuning_objective_does_not_share_the_champion_gates_bootstrap() -> None:
    """v5 live defect 2, as a regression test.

    Both resample a (label, score) vector. Sharing an offset made the search's
    resamples and the promotion gate's resamples the same draw, correlating the
    decision that selects a model with the decision that judges it.
    """
    assert seeds.tuning_objective_bootstrap_seed(7) != seeds.champion_bootstrap_seed(7)
