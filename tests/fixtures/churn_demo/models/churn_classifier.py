"""Hooks for churn_classifier: a custom metric via the escape hatch (FR-RES-07).

Lift and gain are builtins (``lift_at_0.1``); hooks cover what builtins
cannot express - here, churners captured within a fixed 100-contact
retention-campaign budget (count-based, not fraction-based).
"""


def custom_metrics(predictions, ctx):
    scores = predictions.column("prediction").to_pylist()
    labels = predictions.column(ctx.spec.target).to_pylist()
    paired = sorted(zip(scores, labels, strict=False), key=lambda p: p[0], reverse=True)
    captured = sum(label for _, label in paired[:100])
    total = sum(labels)
    return {"campaign_capture_100": (captured / total) if total else 0.0}
