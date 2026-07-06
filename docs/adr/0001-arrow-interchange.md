# ADR-1: Arrow as the data interchange format

**Status:** accepted

## Decision
Data crossing the DataAdapter -> TrainingAdapter boundary is `pyarrow.Table`.

## Rationale
Framework-neutral and zero-copy into XGBoost, LightGBM, Polars, and DuckDB;
keeps ML types out of mbt-core (NFR-04). DuckDB materializes parquet to Arrow
cheaply; adapters convert to their native matrix formats lazily.
