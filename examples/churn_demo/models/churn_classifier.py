"""Hooks for churn_classifier: a custom metric via the escape hatch (FR-RES-07)."""


def custom_metrics(predictions, ctx):
    """Positive rate among the top-scoring decile (a simple lift proxy)."""
    scores = predictions.column("prediction").to_pylist()
    labels = predictions.column(ctx.spec.target).to_pylist()
    if not scores:
        return {"lift_at_decile": 0.0}
    paired = sorted(zip(scores, labels), key=lambda p: p[0], reverse=True)
    top = paired[: max(1, len(paired) // 10)]
    top_rate = sum(label for _, label in top) / len(top)
    base_rate = sum(labels) / len(labels)
    return {"lift_at_decile": (top_rate / base_rate) if base_rate else 0.0}
