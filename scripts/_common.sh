#!/usr/bin/env bash
#
# Shared helpers for the reproduction scripts. SOURCE this file; do not run it.
#
# Conventions used by every scripts/phaseN_*.sh:
#   * They cd to the repo root, so paths in args are repo-relative.
#   * They forward any extra CLI args ("$@") straight to the underlying
#     `python -m experiments...` module, so anything the module accepts works:
#         scripts/phase1_generate.sh --limit 5 --dry-run
#   * Knobs with sensible defaults are read from the environment (SYSTEM,
#     DATA_DIR, CAPACITY, SUBDIR, MAX_WORKERS, PYTHON, ...). See each script's
#     header for the ones it honours.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Interpreter. Default to `python`; set PYTHON=... or run under `uv run` by
# exporting PYTHON="uv run python".
PYTHON="${PYTHON:-python}"

# Where experiment artifacts live (inputs and outputs).
DATA_DIR="${DATA_DIR:-data/experiments}"

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
warn() { printf '\033[1;33m[%s] WARN\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# Echo a command, then run it.
run() {
    log "+ $*"
    "$@"
}

# Run a python module: run_py <module> [args...]
run_py() {
    local module="$1"; shift
    run "$PYTHON" -m "$module" "$@"
}
