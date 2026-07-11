# deploy: the CD source of truth (DESIGN.md sections 2 and 5)

This repo is what continuous delivery reconciles, and nothing else:

- `images.env` - the pinned deployable-unit digest (written by the
  prod-build pipeline's `deploy-digest` step) plus the session wiring the
  Airflow DAGs need (docker network, shared workspace host path).
- `dags/` - the Airflow DAGs. git-sync pulls this repo into the scheduler
  every few seconds; rollback is `git revert`.
- `k8s/` - the same unit rendered as Kubernetes CronJobs for the optional
  k3d + ArgoCD fidelity profile.

Model promotions never touch this repo: champions resolve from the registry
at run time (ADR-20), so the frequent release event is a registry alias
flip. The only thing left to CD is which unit digest the scheduled jobs run.
