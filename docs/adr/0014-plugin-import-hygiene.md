# ADR-14: Plugin modules and param models import no ML framework

**Status:** accepted

`mbt parse` validates tasks and hyperparameters within its 2s budget only
because loading a plugin descriptor is cheap: frameworks import lazily
inside train/evaluate/load/export/predict. The compliance suite enforces
this with a fresh-subprocess sys.modules check.
