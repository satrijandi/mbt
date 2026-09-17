"""The training report (ADR-30): what a training run records beyond its metrics.

Job-side orchestration over the pure numerics in ``mbt_adapter_base.reporting``:
aligning scored splits with their keys and times, cutting the after-test
window into period cells, judging stability against the test split, and
writing the tables and the self-contained HTML page a tracking run carries.
numpy loads lazily inside the job only (ADR-14).
"""
