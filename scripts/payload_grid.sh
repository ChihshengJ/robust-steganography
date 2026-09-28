#!/usr/bin/env bash
#
# The payload grid (ARR October revision): every cell of both tracks, stage by
# stage. A thin wrapper around experiments/payload_grid.py; see its docstring
# for the grid and the stages.
#
#   scripts/payload_grid.sh cells                         # list the 32 cells
#   scripts/payload_grid.sh generate --dry-run            # print the commands
#   scripts/payload_grid.sh generate --track detection --parallel 8
#   scripts/payload_grid.sh select
#   scripts/payload_grid.sh all --systems litreview
#
# Stages: generate, select, normal, attack, decode, analyze (or all). Each one
# resumes; rerun a stage to retry the cells that failed. Per-cell logs go to
# $DATA_DIR/logs/payload_grid/{stage}/.
#
# Env knobs: DATA_DIR, PYTHON, and LOCAL_MODEL/LOCAL_BASE_URL for the pinned G.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_py experiments.payload_grid --data-dir "$DATA_DIR" "$@"
