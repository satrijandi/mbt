#!/bin/sh
# What upstream's rootless docker-setup.sh does, minus its templating: put
# the stock app.ini in place on first start, then fold the GITEA__section__KEY
# environment (compose's configuration) into it on every start.
#
# The config lives where Gitea looks by default, $GITEA_CUSTOM/conf/app.ini,
# so `docker exec <gitea> gitea admin ...` (ci_bootstrap.py) finds it with no
# wrapper and no --config.
set -eu

conf="$GITEA_CUSTOM/conf/app.ini"
mkdir -p "$HOME" "$GITEA_TEMP" "$(dirname "$conf")"
chmod 0700 "$HOME" "$GITEA_TEMP" "$GITEA_CUSTOM"
[ -f "$conf" ] || cp /usr/local/share/gitea/app.ini "$conf"
gitea config edit-ini --in-place --apply-env --config "$conf"

exec "$@"
