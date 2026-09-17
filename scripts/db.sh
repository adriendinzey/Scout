#!/usr/bin/env bash
#
# Docker Compose wrapper that loads this sandbox's isolated database settings.
#
# In the main clone it is a plain passthrough to `docker compose`. Inside a
# worktree created by scripts/worktree.sh it first sources .sandbox/env, so the
# sandbox's own port and Compose project name apply — which is what keeps
# parallel tasks from fighting over port 5433 and one shared pgdata volume.
#
# Usage:
#   scripts/db.sh up            start (and wait for) Postgres
#   scripts/db.sh down          stop it
#   scripts/db.sh psql          open a shell on it
#   scripts/db.sh <any compose subcommand>

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$here/.sandbox/env" ]]; then
  # shellcheck disable=SC1091
  . "$here/.sandbox/env"
fi

cd "$here"

case "${1:-}" in
  up)
    shift
    # --wait blocks on the healthcheck, which verifies BOTH extensions loaded —
    # not merely that the port is open.
    exec docker compose up -d --wait "$@"
    ;;
  psql)
    shift
    exec docker compose exec db psql -U scout -d scout "$@"
    ;;
  "")
    exec docker compose ps
    ;;
  *)
    exec docker compose "$@"
    ;;
esac
