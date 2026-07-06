# ADR-2: Local Data/Compute adapters ship inside mbt-core

**Status:** accepted

## Decision
The `local` plugin (Parquet/DuckDB DataAdapter + subprocess ComputeAdapter)
lives in mbt-core and registers through the same entry-point mechanism as
external adapters.

## Rationale
Batteries included for the one-hour quickstart (G5). DuckDB/PyArrow are data
dependencies, not ML frameworks, so the "no ML deps in core" rule (NFR-04)
holds. Dogfooding the plugin mechanism from inside core keeps the contract
honest.
