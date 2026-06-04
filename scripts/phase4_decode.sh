#!/usr/bin/env bash
#
# Phase 4 — decode attacked stegos (BER) and score attack severity, then
# aggregate the main results tables.
#
#   4a decode           : API + local llama.cpp server for TopicQA/Story decode
#   4b attack metrics    : BERTScore/cosine/BLEU/TER (gpu/mps/cpu, no API)
#   4c main results      : aggregate into paper tables (no API)
#
# 4a/4b run per system against {system}_cap{N}/ (pass the SAME CAPACITY used in
# Phases 1 and 3). 4c reads all systems at once via --capacities and is invoked
# separately by run_all.sh / run_robustness.sh — not here.
#     SYSTEM=topicqa CAPACITY=6 scripts/phase4_decode.sh
#
# Env knobs: SYSTEM (default all*), CAPACITY, SUBDIR, DATA_DIR, PYTHON.
# Extra args are forwarded to phase4a_decode.
# *CAPACITY requires SYSTEM != all; loop per system for the native-capacity run.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SYSTEM="${SYSTEM:-all}"

args=( --system "$SYSTEM" --data-dir "$DATA_DIR" )
[ -n "${CAPACITY:-}" ] && args+=( --capacity "$CAPACITY" )
[ -n "${SUBDIR+x}" ]   && args+=( --subdir "$SUBDIR" )

run_py experiments.phase4_decode.phase4a_decode         "${args[@]}" "$@"
run_py experiments.phase4_decode.phase4b_attack_metrics "${args[@]}"
