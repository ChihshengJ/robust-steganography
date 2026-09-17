#!/usr/bin/env bash
#
# Length-matched Discop baseline with SyncPool — Phases 1, 3 and 4 for one
# capacity, against the same subdir throughout.
#
# This exists because the run is four commands that must agree on a subdir the
# phase modules do not all derive the same way: Phase 1 auto-names
# '{system}_cap{N}_len{T}_sp', while Phases 3/4 only auto-name
# '{system}_cap{N}'. Getting that wrong reads the wrong Phase 1 records (or,
# worse, silently finds none). The subdir is computed once here and passed to
# every phase.
#
# Run the clean-channel gate FIRST — it is the precondition for the whole thing
# (ECC_AND_RATE_PLAN.md §5/§6):
#     scripts/baselines_syncpool.sh --gate-only
#
# Then, per capacity:
#     CAPACITY=14 scripts/baselines_syncpool.sh
#     CAPACITY=16 scripts/baselines_syncpool.sh
#
# Env knobs:
#   CAPACITY      payload bits, e.g. 14      (default 16)
#   TARGET_WORDS  length-matching target     (default 575)
#   REPETITIONS   skip calibration and pin r (default: calibrate)
#   PHASES        which to run, e.g. "1 3 4" (default "1 3 4")
#   MAX_WORKERS   Phase 3 parallelism        (default 8)
#   DECODE_WORKERS Phase 4a parallelism       (default: performance cores)
#   DATA_DIR, PYTHON                         (see _common.sh)
#
# STEGO ONLY. This script produces the *recovery* curve, and only stego texts
# carry bits. C1 (same-pipeline cover) and C2 (prompted GPT-4.1 cover) exist for
# the steganalysis/detection experiment; here they are pure cost — C1 is a second
# full GPT-2 encode per prompt and C2 is an API call — so Phase 1 runs
# --stego-only and Phase 3 --skip-covers. That roughly halves Phase 1, removes
# its API dependency entirely, and drops Phase 3 from 850 attacked records
# to 810.
#
# Extra args are forwarded to Phase 1 only (e.g. --limit 2 for a smoke run).
#
# NOTE ON COST. Decoding is a GPT-2 forward per token, on CPU: the backend uses
# float64 arithmetic that MPS does not support, and encode/decode must agree
# bit-for-bit or the stream desyncs, so there is no device shortcut. Measured
# here: 254 ms per step for gpt2-medium single-threaded, which is >99% of the
# per-step cost (the pooling, the Huffman build and the KV-cache crop are all
# under 0.5 ms combined). Phase 4 dominates; DECODE_WORKERS is the knob.
#
# Attacked text still runs longer than clean text per document -- the byte walk
# desyncs and consumes shorter tokens, ~2.3 bytes/step against a clean ~4.3 --
# but the decoder now stops at `error_encoded_length` bits instead of at the
# last byte, since everything past that is sliced off unread. That cuts an
# attacked decode from ~1465 steps to ~540. Measured figures in
# ECC_AND_RATE_PLAN.md §9.6.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# GPT-2 size for the baseline. Medium, not small, and this is a measured
# choice rather than a preference: at the length-matched rate the encoder has to
# produce ~750 tokens, and gpt2-small degenerates into repetition loops well
# before that. Over 5 pilots x 800 steps, small spent 28.5% of steps with a
# single-candidate nucleus (zero bits embedded, no escape — 2 of 5 trajectories
# collapsed outright), medium 9.5% with none collapsing. Medium also embeds 57%
# more per token (2.62 vs 1.67 b/token), so a larger r fits the same 575 words
# and the baseline gets *more* redundancy, not less.
#
# It is exported, not hardcoded in the factory, so the existing gpt2 datasets
# keep decoding under gpt2: Phase 4 takes the model from each record's Phase 1
# metadata, and this only sets what *new* records are generated with.
export BASELINE_MODEL="${BASELINE_MODEL:-gpt2-medium}"

CAPACITY="${CAPACITY:-16}"
TARGET_WORDS="${TARGET_WORDS:-575}"
PHASES="${PHASES:-1 3 4}"
MAX_WORKERS="${MAX_WORKERS:-8}"
# Phase 4 decode parallelism: one process per record, single-threaded each.
# Phase 4 is the whole cost of this pipeline (810 attacked decodes x a forward
# per token) and threads do nothing for it, so this is the knob that matters.
# Default to the performance-core count; each worker holds its own LM copy.
DECODE_WORKERS="${DECODE_WORKERS:-$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo 4)}"

if [ "${1:-}" = "--gate-only" ]; then
    run_py experiments.utils.syncpool_gate
    exit $?
fi

SYSTEM=discop

# Must match phase1_generate's auto-subdir rule exactly.
SUBDIR="${SYSTEM}_cap${CAPACITY}_len${TARGET_WORDS}_sp"
log "system=$SYSTEM capacity=$CAPACITY model=$BASELINE_MODEL subdir=$SUBDIR phases='$PHASES'"

has_phase() { [[ " $PHASES " == *" $1 "* ]]; }

if has_phase 1; then
    p1=( --system "$SYSTEM" --data-dir "$DATA_DIR" --capacity "$CAPACITY"
         --subdir "$SUBDIR" --syncpool --length-matched
         --target-words "$TARGET_WORDS" --stego-only )
    [ -n "${REPETITIONS:-}" ] && p1+=( --repetitions "$REPETITIONS" )
    run_py experiments.phase1_generation.phase1_generate "${p1[@]}" "$@"
fi

if has_phase 3; then
    run_py experiments.phase3_attacks \
        --system "$SYSTEM" --data-dir "$DATA_DIR" --subdir "$SUBDIR" \
        --max-workers "$MAX_WORKERS" --skip-covers
fi

if has_phase 4; then
    run_py experiments.phase4_decode.phase4a_decode \
        --system "$SYSTEM" --data-dir "$DATA_DIR" --subdir "$SUBDIR" \
        --max-workers "$DECODE_WORKERS"
    run_py experiments.phase4_decode.phase4b_attack_metrics \
        --system "$SYSTEM" --data-dir "$DATA_DIR" --subdir "$SUBDIR"
fi

log "done: $SUBDIR"
