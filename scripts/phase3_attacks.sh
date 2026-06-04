#!/usr/bin/env bash
#
# Phase 3 — apply the 5 attacks (synonym, local/global paraphrase,
# local/global back-translation) to the first 30 stegos + 20 covers per system.
# API-heavy (paraphrase/translation call OpenAI); the local llama.cpp server is
# NOT needed here. The synonym attack needs NLTK WordNet (see README setup).
#
# Reads the robustness dataset produced by `CAPACITY=N scripts/phase1_generate.sh`.
# Pass the SAME CAPACITY so it reads/writes the matching {system}_cap{N}/ subdir:
#     SYSTEM=topicqa CAPACITY=6  scripts/phase3_attacks.sh
#
# Env knobs: SYSTEM (default all*), CAPACITY, SUBDIR, MAX_WORKERS (default 8),
#            DATA_DIR, PYTHON. Extra args forwarded, e.g. --attack global_paraphrase.
# *CAPACITY requires SYSTEM != all; loop per system for the native-capacity run.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SYSTEM="${SYSTEM:-all}"
MAX_WORKERS="${MAX_WORKERS:-8}"

args=( --system "$SYSTEM" --data-dir "$DATA_DIR" --max-workers "$MAX_WORKERS" )
[ -n "${CAPACITY:-}" ] && args+=( --capacity "$CAPACITY" )
[ -n "${SUBDIR+x}" ]   && args+=( --subdir "$SUBDIR" )

run_py experiments.phase3_attacks "${args[@]}" "$@"
