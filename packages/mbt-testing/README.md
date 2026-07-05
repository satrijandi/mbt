# mbt-testing

Fake, contract-conformant adapters for testing mbt projects (and mbt itself)
without ML frameworks or external services.

The `fake` plugin provides:

- **training**: a deterministic "model" whose metrics are controlled by the
  `fake_metric_value` hyperparameter - ideal for exercising gates.
- **tracking / registry**: file-backed under the project's `target/`, so
  assertions work across the coordinator/job process boundary.
- **tuning**: a seeded random sampler honoring trial caps.
- **compute**: an *inline* compute adapter running jobs in-process
  (fast tests, easy debugging); use the built-in `local` adapter for real
  subprocess isolation.
