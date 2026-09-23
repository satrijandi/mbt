"""The half of a training adapter that is not per-framework (A-4).

Normalising only the framework name, 87 of the LightGBM adapter's 173
substantive lines appeared verbatim in the XGBoost adapter. ``evaluate`` was
identical in three packages but for a type annotation; so was ``predict``;
``_fit_calibrator`` recurred in all five, each with a near-verbatim docstring
saying it was "the same mechanism as the xgboost adapter".

The surface an adapter author had to learn was 22 named things - 5 properties,
9 required methods, 4 optional protocols, 3 undeclared ``supports_*`` class
variables and an undeclared ``__init__(config: dict)`` convention - while only
four things actually varied: ``train``, ``_scores``, ``export``, ``load``.

``ArrowTrainingAdapter`` keys everything else on the one abstract hook every
Arrow adapter already had, ``_scores(model, table)``.

**Scope.** The three Arrow adapters (XGBoost, LightGBM, scikit-learn) only.
h2o and Spark take ``(model, data, split)`` because of the ``data_access``
path/arrow divide (ADR-17), so they implement the protocol directly and share
``capabilities`` through ``method_capabilities`` instead. That is the sequencing
A-4 asked for, not an oversight.
"""

from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from mbt_adapter_base.capabilities import Capability, method_capabilities
from mbt_adapter_base.interchange import MetricResults

if TYPE_CHECKING:
    import numpy as np

    from mbt_adapter_base.protocols import DatasetHandle
    from mbt_adapter_base.specs import MetricSpec, ModelSpec


class ArrowTrainingAdapter:
    """Shared behaviour for adapters that read splits as Arrow tables.

    A subclass implements ``train``, ``_scores``, ``export`` and ``load``, plus
    the descriptive properties. Everything here is derived from ``_scores``.
    """

    #: Set by the subclass; ``data_access`` is "arrow" for everything here.
    data_access: ClassVar[str] = "arrow"

    #: Which optional capabilities this adapter has beyond the ones implied by
    #: the methods it defines. Subclasses that can post-hoc calibrate, honour
    #: monotone constraints or pool rare levels list them here; one that needs
    #: to decide per spec overrides ``capabilities`` instead.
    extra_capabilities: ClassVar[frozenset[Capability]] = frozenset()

    def capabilities(self, spec: "ModelSpec | None" = None) -> frozenset[Capability]:
        """What this adapter can do, for ``spec`` when one is given (B-1).

        The default is "whichever optional methods this class defines, plus
        ``extra_capabilities``". An adapter whose answer depends on the spec -
        sklearn, whose monotone support is a property of the ESTIMATOR rather
        than of the library - overrides this.
        """
        return method_capabilities(self) | self.extra_capabilities

    # -- derived from _scores --------------------------------------------------

    def _scores(self, model: Any, table: pa.Table) -> "np.ndarray":
        """Model scores for one table: the one thing every subclass must have.

        Post-hoc calibration belongs here, so ``evaluate``, ``predict`` and the
        paired champion delta all see calibrated probabilities.
        """
        raise NotImplementedError

    def evaluate(
        self,
        model: Any,
        data: "DatasetHandle",
        split: str,
        metrics: "list[MetricSpec]",
        slices: list[str] | None = None,
    ) -> MetricResults:
        from mbt_adapter_base.training_helpers import evaluate_split

        table = data.read(split)
        return evaluate_split(table, model.target, self._scores(model, table), metrics, slices)

    def predict(self, model: Any, data: "DatasetHandle", split: str) -> pa.Table:
        """The split's table plus a ``prediction`` column."""
        table = data.read(split)
        scores = self._scores(model, table)
        return table.append_column("prediction", pa.array(scores.astype("float64")))

    def fit_calibrator(self, model: Any, spec: "ModelSpec", data: "DatasetHandle") -> Any:
        """Fit a post-hoc probability calibrator and return it (R2-8, F17).

        The calibrator must fit on rows the model never trained on and that no
        selection step optimized against, so it uses the dedicated
        ``calibration`` slice core carves from train, falling back to
        ``validation`` for direct callers. Without either there is no honest
        calibration set and ``calibration_split`` fails loudly.

        Attaching it to the model is per-framework (XGBoost persists it as a
        booster attribute, LightGBM and sklearn hold it on the wrapper), so the
        subclass does that in ``_fit_calibrator``; the FITTING is here, and it
        was written five times.
        """
        from mbt_adapter_base.calibration import Calibrator
        from mbt_adapter_base.training_helpers import calibration_split

        assert spec.calibration is not None  # guarded by the caller
        table = data.read(calibration_split(data))
        raw = self._scores(model, table)  # no calibrator attached yet -> raw
        labels = table.column(spec.target).to_numpy(zero_copy_only=False)
        return Calibrator.fit(raw, labels, spec.calibration)


class ShapArrowTrainingAdapter(ArrowTrainingAdapter):
    """An Arrow adapter whose framework can produce SHAP contributions.

    SHAP lives on its OWN base rather than on ``ArrowTrainingAdapter``, because
    capability is derived from which methods a class has (B-1): putting
    ``shap_importance`` on the shared base would have made scikit-learn - which
    cannot produce contributions - declare the capability and then raise
    ``NotImplementedError`` when core took it at its word. Inheriting a method
    you cannot honour is exactly the failure the capability rewrite exists to
    prevent, so the inheritance says what is true.

    A subclass implements ``_shap_values``; both public methods derive from it.
    """

    def _shap_values(self, model: Any, table: pa.Table) -> "np.ndarray":
        """Per-feature SHAP contributions ``[n_rows, n_features]``.

        The trailing bias/base-value column the frameworks append is the
        subclass's to drop.
        """
        raise NotImplementedError

    def shap_importance(self, model: Any, data: "DatasetHandle", split: str) -> dict[str, float]:
        """Global importance as mean |SHAP| over the split, as fractions.

        Preferred over gain in the model card: SHAP contributions are additive
        and, unlike split-gain, not biased toward high-cardinality features, so
        they rank a many-valued column against a binary one fairly.
        """
        import numpy as np

        mean_abs = np.abs(self._shap_values(model, data.read(split))).mean(axis=0)
        total = float(mean_abs.sum()) or 1.0  # a model that learned nothing -> all zeros
        return {
            name: round(float(value) / total, 6)
            for name, value in zip(model.features, mean_abs, strict=True)
        }

    def explain(self, model: Any, data: "DatasetHandle", split: str, top_k: int) -> list[str]:
        """Per-row local attribution: the ``top_k`` features by |SHAP| per row."""
        from mbt_adapter_base.training_helpers import top_k_explanations

        return top_k_explanations(self._shap_values(model, data.read(split)), model.features, top_k)


__all__ = ["ArrowTrainingAdapter", "ShapArrowTrainingAdapter"]
