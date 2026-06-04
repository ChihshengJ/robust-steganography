#!/usr/bin/env bash
#
# Phase 1 — generate stego (S), same-pipeline cover (C1) and prompted cover (C2)
# texts. API-heavy. TopicQA and Story additionally need the local llama.cpp
# server up (scripts/serve_local_model.sh); LitReview and baseline do not.
#
# Two layouts, selected by how you call it:
#
#   * Steganalysis dataset (full 300/class at native capacity, top-level
#     phase1_texts/ — this is what Phase 2 reads):
#         SYSTEM=topicqa SUBDIR='' scripts/phase1_generate.sh
#
#   * Robustness dataset (per-system native capacity in {system}_cap{N}/ —
#     what Phase 3/4 read). Pass CAPACITY; the module auto-names the subdir:
#         SYSTEM=topicqa CAPACITY=6  scripts/phase1_generate.sh
#         SYSTEM=story   CAPACITY=18 scripts/phase1_generate.sh   # 20 slots, +2 convention
#         SYSTEM=litreview CAPACITY=20 scripts/phase1_generate.sh
#
# Env knobs: SYSTEM (default all), CAPACITY, SUBDIR, DATA_DIR, PYTHON.
# Anything else is forwarded, e.g.:  scripts/phase1_generate.sh --limit 5 --dry-run
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SYSTEM="${SYSTEM:-all}"

args=(--system "$SYSTEM" --data-dir "$DATA_DIR")
[ -n "${CAPACITY:-}" ] && args+=(--capacity "$CAPACITY")
# SUBDIR is honoured even when set to the empty string (selects top-level).
[ -n "${SUBDIR+x}" ] && args+=(--subdir "$SUBDIR")

run_py experiments.phase1_generation.phase1_generate "${args[@]}" "$@"
