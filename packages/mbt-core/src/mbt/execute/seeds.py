"""The seed ladder: every seeded stage of a training run, in one place (B-3).

A model spec declares one ``seed``. Every stage of a run that needs randomness
derives its own seed from it by a fixed offset, so a run is reproducible as a
whole while no two stages share a random stream. ADR-18 named the first rungs
and ADR-30 added the seventh; before this module the ladder existed only as
inline arithmetic (``spec.seed + 3``) documented in five prose locations, none
of them executable - and ``+3`` was taken twice, by the champion-gate bootstrap
and by the tuning-time robust objective.

So the ladder lives here, as a registry rather than as a convention:

- ``SeedRung`` is the whole ladder, readable in one screen.
- one named function per rung is what call sites use, so the arithmetic never
  appears in ``job.py`` again.
- ``test_seeds_unit.py`` asserts the offsets are distinct, which is what makes
  a second ``+3`` a test failure rather than a silent correlation.

**Adding a stage**: add a rung with the next free offset, add its accessor, and
use it. Never reuse an offset, and never renumber an existing one - the rung
numbers are part of a run's reproducibility, so changing one silently changes
results for every project that pinned a seed.
"""

from enum import IntEnum

__all__ = [
    "SeedRung",
    "calibration_carve_seed",
    "champion_bootstrap_seed",
    "operating_point_carve_seed",
    "permutation_importance_seed",
    "random_kfold_seed",
    "seed_for",
    "train_seed",
    "tuning_objective_bootstrap_seed",
    "tuning_seed",
    "validation_carve_seed",
]


class SeedRung(IntEnum):
    """Offset from ``spec.seed`` for each seeded stage of a run.

    The values are a contract: a project that pins ``seed: 42`` expects the
    same rows carved and the same folds drawn on the next run.
    """

    #: The model fit itself - the seed handed to the framework (ADR-18).
    TRAIN = 0
    #: The tuning engine's sampler (which trials it proposes).
    TUNING = 1
    #: The implicit validation carve taken out of train (TSD §13.5).
    VALIDATION_CARVE = 2
    #: The champion gate's paired bootstrap resamples (ADR-18).
    CHAMPION_BOOTSTRAP = 3
    #: Fold assignment for the random-split backtest (k-fold).
    RANDOM_KFOLD = 4
    #: The dedicated calibration slice carved out of train (F17).
    CALIBRATION_CARVE = 5
    #: Row sample for the permutation-importance fallback (ADR-30).
    PERMUTATION_IMPORTANCE = 6
    #: The tuning objective's own bootstrap resamples (R2-7). Distinct from
    #: CHAMPION_BOOTSTRAP: both resample a (label, score) vector, and sharing
    #: an offset made the search's resamples and the promotion gate's
    #: resamples the same draw - correlating the decision that selects a model
    #: with the decision that judges it. They were the same rung until v5.
    TUNING_OBJECTIVE_BOOTSTRAP = 7
    #: The operating-point carve: rows a deployable cutoff is selected on, held
    #: out from the rows that report its precision (D-1).
    OPERATING_POINT_CARVE = 8


def seed_for(base: int, rung: SeedRung) -> int:
    """The seed one stage runs at, given the spec's declared ``seed``."""
    return base + int(rung)


def train_seed(base: int) -> int:
    """The model fit's own seed."""
    return seed_for(base, SeedRung.TRAIN)


def tuning_seed(base: int) -> int:
    """The tuning engine's sampler seed."""
    return seed_for(base, SeedRung.TUNING)


def validation_carve_seed(base: int) -> int:
    """The implicit validation carve's seed."""
    return seed_for(base, SeedRung.VALIDATION_CARVE)


def champion_bootstrap_seed(base: int) -> int:
    """The champion gate's paired-bootstrap seed (ADR-18)."""
    return seed_for(base, SeedRung.CHAMPION_BOOTSTRAP)


def random_kfold_seed(base: int) -> int:
    """The random-split backtest's fold-assignment seed."""
    return seed_for(base, SeedRung.RANDOM_KFOLD)


def calibration_carve_seed(base: int) -> int:
    """The calibration slice's carve seed (F17)."""
    return seed_for(base, SeedRung.CALIBRATION_CARVE)


def permutation_importance_seed(base: int) -> int:
    """The permutation-importance row sample's seed (ADR-30)."""
    return seed_for(base, SeedRung.PERMUTATION_IMPORTANCE)


def tuning_objective_bootstrap_seed(base: int) -> int:
    """The tuning objective's bootstrap seed (R2-7)."""
    return seed_for(base, SeedRung.TUNING_OBJECTIVE_BOOTSTRAP)


def operating_point_carve_seed(base: int) -> int:
    """The operating-point carve's seed (D-1)."""
    return seed_for(base, SeedRung.OPERATING_POINT_CARVE)
