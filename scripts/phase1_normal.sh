#!/usr/bin/env bash
#
# Phase 1b — length-matched normal generations for steganalysis. API-heavy; no
# local server needed. Run after phase1_generate.sh has written the stegotexts.
# One per stegotext, by its own synthesizer at its own sampling, written next
# to it as {dir}/{system}_normal.jsonl.
#
#     SYSTEM=story DIRS='story_cap*' scripts/phase1_normal.sh
#
# Env knobs: SYSTEM (story|litreview, required), DIRS (space-separated globs
# under phase1_texts/, default '{SYSTEM}_cap*'), WORKERS (default 8), DATA_DIR,
# PYTHON. Anything else is forwarded, e.g.:  scripts/phase1_normal.sh --dry-run
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

: "${SYSTEM:?set SYSTEM=story or SYSTEM=litreview}"
DIRS="${DIRS:-${SYSTEM}_cap*}"

# shellcheck disable=SC2206  # DIRS is intentionally word-split, not globbed here
set -f
dirs=(${DIRS})
set +f

run_py experiments.phase1_generation.phase1_normal \
    --system "$SYSTEM" --data-dir "$DATA_DIR" --dirs "${dirs[@]}" \
    --workers "${WORKERS:-8}" "$@"
