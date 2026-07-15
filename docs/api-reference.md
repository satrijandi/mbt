# Adapter API reference

Generated from `mbt-adapter-base`, the only package an adapter may import
(the compliance suite's `test_no_core_imports` enforces it).
Prose guide: [Adapter authoring](adapter-authoring.md).

## Protocols

What an adapter implements.
The two `Supports*` protocols are optional capabilities probed with `hasattr`; implementing them with these exact signatures is what keeps adapters from drifting apart.

::: mbt_adapter_base.protocols
    options:
      show_source: false
      members_order: source

## Interchange types

Everything that crosses the adapter boundary is one of these serializable types.

::: mbt_adapter_base.interchange
    options:
      show_source: false
      members_order: source

## Shared metric engine

::: mbt_adapter_base.metrics
    options:
      show_source: false
      filters: ["!^_"]

## Shared training helpers

::: mbt_adapter_base.training_helpers
    options:
      show_source: false

## Compliance suite

Subclass these in your adapter's tests; passing them is the ship bar.

::: mbt_adapter_base.compliance.suite
    options:
      show_source: false
      filters: ["!^_"]
