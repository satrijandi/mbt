"""Post-hoc probability calibration (R2-8).

A calibrator maps a binary classifier's raw scores to calibrated probabilities,
fit on a held-out split. It is a small, serializable transform - NOT a model -
so an adapter persists it beside the trained model (e.g. as a booster attribute)
and applies it wherever scores are produced; both champion and challenger carry
their own, so a paired gate comparison stays apples-to-apples. Fitting is
deterministic given the data, preserving mbt's reproducibility guarantee.

Two methods:
- ``isotonic``: a monotonic, non-parametric step fit (sklearn IsotonicRegression);
  flexible, needs enough held-out rows to be stable.
- ``sigmoid``: Platt scaling, a two-parameter logistic ``1/(1+exp(-(a*s+b)))``;
  robust on small held-out sets, assumes a sigmoidal distortion.

Transform is pure numpy (no sklearn at score time), so a loaded champion applies
its calibrator without depending on the fitting library's serialization format.
"""

import json
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np

CalibrationMethod = Literal["isotonic", "sigmoid"]


class Calibrator:
    """A fitted score-to-probability transform (``method`` + serializable params)."""

    def __init__(self, method: CalibrationMethod, params: dict[str, Any]) -> None:
        self.method = method
        self.params = params

    @classmethod
    def fit(cls, scores: Any, y_true: Any, method: CalibrationMethod) -> "Calibrator":
        import numpy as np

        x = np.asarray(scores, dtype=float)
        y = np.asarray(y_true, dtype=float)
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip").fit(x, y)
            return cls(
                "isotonic",
                {
                    "x": [float(v) for v in iso.X_thresholds_],
                    "y": [float(v) for v in iso.y_thresholds_],
                },
            )
        from sklearn.linear_model import LogisticRegression

        # Platt scaling WITH target smoothing (Platt 1999): a near-unregularized
        # logistic on the raw 0/1 labels overfits a small held-out calibration
        # set, so pull the targets toward the prior - t+ = (N+ + 1)/(N+ + 2),
        # t- = 1/(N- + 2). LogisticRegression fits class labels, so present each
        # score as BOTH classes weighted by t and 1 - t: the weighted cross-entropy
        # is identical to fitting the soft target, using only the public API (no
        # private _sigmoid_calibration, no hand-rolled Newton). This also makes a
        # single-class calibration split well-posed, where the raw fit would raise.
        n_pos = float(np.count_nonzero(y >= 0.5))
        n_neg = float(len(y) - n_pos)
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        targets = np.where(y >= 0.5, t_pos, t_neg)
        features = np.concatenate([x, x]).reshape(-1, 1)
        classes = np.concatenate([np.ones(len(x)), np.zeros(len(x))])
        weights = np.concatenate([targets, 1.0 - targets])
        lr = LogisticRegression(C=1e10, solver="lbfgs").fit(
            features, classes, sample_weight=weights
        )
        return cls("sigmoid", {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])})

    def transform(self, scores: Any) -> "np.ndarray":
        import numpy as np

        x = np.asarray(scores, dtype=float)
        if self.method == "isotonic":
            xs = np.asarray(self.params["x"], dtype=float)
            ys = np.asarray(self.params["y"], dtype=float)
            # np.interp with endpoint clamping is exactly IsotonicRegression's
            # out_of_bounds="clip" behaviour, without needing sklearn at load time.
            return np.asarray(np.interp(x, xs, ys), dtype=float)
        a, b = float(self.params["a"]), float(self.params["b"])
        return 1.0 / (1.0 + np.exp(-(a * x + b)))

    def to_json(self) -> str:
        return json.dumps({"method": self.method, "params": self.params}, sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> "Calibrator":
        payload = json.loads(blob)
        return cls(payload["method"], payload["params"])
