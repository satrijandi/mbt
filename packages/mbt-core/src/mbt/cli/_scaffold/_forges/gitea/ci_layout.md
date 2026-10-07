- `.woodpecker/` - Woodpecker CI pipelines for this Gitea (or Forgejo) repo:
  PR check, prod build, promotion, weekly + monthly retrain, daily scoring,
  weekly ground-truth monitor; `scripts/gitea_pr_comment.py` posts the PR
  report, `scripts/ci_git_remote.sh` and `scripts/ci_notify.sh` are their
  shared plumbing
