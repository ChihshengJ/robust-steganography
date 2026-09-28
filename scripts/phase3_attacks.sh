#!/usr/bin/env bash
#
# Phase 3 — apply the attacks (synonym, local paraphrase at p = 0.5/1.0, global
# paraphrase, global round-trip translation through Japanese) to the first 30
# stegos per cell. API-heavy (paraphrase/translation call the attacker); the local llama.cpp
# server is NOT needed here. The synonym attack needs NLTK WordNet (see README).
#
# Reads the recovery cells produced by `TRACK=recovery CAPACITY=N
# scripts/phase1_generate.sh`. Pass the SAME CAPACITY and TRACK so it reads and
# writes the track's one cell at that F:
#     SYSTEM=story CAPACITY=16 TRACK=recovery scripts/phase3_attacks.sh
#
# Env knobs: SYSTEM (default all*), CAPACITY, TRACK, SUBDIR, MAX_WORKERS (default 8),
#            ATTACKER_MODEL (default zai-org/GLM-5.3-Flash),
#            ATTACKER_PROVIDER (together|openai, default together),
#            DATA_DIR, PYTHON. Extra args forwarded, e.g. --attack global_paraphrase.
# Another attacker writes into the same attacked file under its own ids:
#     SYSTEM=story CAPACITY=16 ATTACKER_PROVIDER=together \
#         ATTACKER_MODEL=deepseek-ai/DeepSeek-V4-Flash scripts/phase3_attacks.sh
# *CAPACITY requires SYSTEM != all; loop per system for the native-capacity run.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SYSTEM="${SYSTEM:-all}"
MAX_WORKERS="${MAX_WORKERS:-8}"

args=( --system "$SYSTEM" --data-dir "$DATA_DIR" --max-workers "$MAX_WORKERS" )
[ -n "${CAPACITY:-}" ] && args+=( --capacity "$CAPACITY" )
[ -n "${TRACK:-}" ]    && args+=( --track "$TRACK" )
[ -n "${SUBDIR+x}" ]   && args+=( --subdir "$SUBDIR" )
[ -n "${ATTACKER_MODEL:-}" ]    && args+=( --attacker-model "$ATTACKER_MODEL" )
[ -n "${ATTACKER_PROVIDER:-}" ] && args+=( --attacker-provider "$ATTACKER_PROVIDER" )

run_py experiments.phase3_attacks "${args[@]}" "$@"
