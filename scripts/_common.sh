#!/usr/bin/env bash
#
# Shared helpers for the reproduction scripts. SOURCE this file; do not run it.
#
# Conventions used by every scripts/phaseN_*.sh:
#   * They cd to the repo root, so paths in args are repo-relative.
#   * They forward any extra CLI args ("$@") straight to the underlying
#     `python -m experiments...` module, so anything the module accepts works:
#         scripts/phase1_generate.sh --limit 5 --dry-run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Interpreter, held as an array so a multi-word command (e.g. `uv run python`)
# is executed as separate words rather than one program literally named
# "uv run python". Precedence:
#   1. PYTHON env var, if set (word-split, so PYTHON="uv run python" works);
#   2. `uv run python` when uv is on PATH (the project's venv, no activation
#      needed) — this is the common case after `uv sync`;
#   3. plain `python` otherwise.
if [ -n "${PYTHON:-}" ]; then
    # shellcheck disable=SC2206  # intentional word-split of a user-set command
    PYTHON=(${PYTHON})
elif command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
else
    PYTHON=(python)
fi

# Where experiment artifacts live (inputs and outputs).
DATA_DIR="${DATA_DIR:-data/experiments}"

log() { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
warn() { printf '\033[1;33m[%s] WARN\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# Echo a command, then run it.
run() {
    log "+ $*"
    "$@"
}

# Run a python module: run_py <module> [args...]
run_py() {
    local module="$1"
    shift
    run "${PYTHON[@]}" -m "$module" "$@"
}
