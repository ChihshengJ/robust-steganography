#!/usr/bin/env bash
#
# End-to-end reproduction of the paper's two data tracks at NATIVE capacity.
# This is the full, API-heavy run (hundreds–thousands of OpenAI calls + GPU
# work). Use scripts/smoke_test.sh first to validate your setup for free.
#
#   scripts/run_all.sh [detectability|robustness|all]   (default: all)
#
# Prerequisites:
#   * .env with OPENAI_API_KEY (see .env.example).
#   * For TopicQA/Story: the local llama.cpp server running in another terminal
#     (scripts/serve_local_model.sh /path/to/model.gguf). LitReview/baseline
#     don't need it.
#
# Tracks:
#   detectability — full 300/class top-level dataset -> Phase 2 (Exp 2 & 5).
#   robustness    — per-system native-capacity dataset -> Phase 3/4 (Exp 1 & 3).
#
# Env knobs:
#   DETECT_SYSTEMS  systems for the detectability track (default topicqa story litreview)
#   ROBUST_SYSTEMS  "name:capacity" list (default native: topicqa:6 story:18 litreview:20 baseline:3)
#   OUTPUT_SUBDIR   phase4c output dir name (default main_native)
#   DATA_DIR, PYTHON, MAX_WORKERS, and the per-phase knobs all pass through.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SCRIPTS="$REPO_ROOT/scripts"

TRACK="${1:-all}"
DETECT_SYSTEMS="${DETECT_SYSTEMS:-topicqa story litreview}"
ROBUST_SYSTEMS="${ROBUST_SYSTEMS:-topicqa:6 story:18 litreview:20 baseline:3}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-main_native}"

run_detectability() {
    log "=== Track: detectability (Phase 1 top-level + Phase 2) ==="
    for sys in $DETECT_SYSTEMS; do
        SYSTEM="$sys" SUBDIR='' "$SCRIPTS/phase1_generate.sh"
    done
    SYSTEMS="$(echo "$DETECT_SYSTEMS" | tr ' ' ',')" "$SCRIPTS/phase2_metrics.sh"
}

run_robustness() {
    log "=== Track: robustness (Phase 1 per-capacity + Phase 3/4) ==="
    local caps="" syslist=""
    for entry in $ROBUST_SYSTEMS; do
        local sys="${entry%%:*}" cap="${entry##*:}"
        SYSTEM="$sys" CAPACITY="$cap" "$SCRIPTS/phase1_generate.sh"
        SYSTEM="$sys" CAPACITY="$cap" "$SCRIPTS/phase3_attacks.sh"
        SYSTEM="$sys" CAPACITY="$cap" "$SCRIPTS/phase4_decode.sh"
        caps="${caps:+$caps,}$sys=$cap"
        syslist="${syslist:+$syslist,}$sys"
    done
    log "=== Phase 4c: aggregate main results ($caps) ==="
    run_py experiments.phase4_decode.phase4c_main_results \
        --data-dir "$DATA_DIR" \
        --capacities "$caps" \
        --systems "$syslist" \
        --output-subdir "$OUTPUT_SUBDIR"
}

case "$TRACK" in
    detectability) run_detectability ;;
    robustness)    run_robustness ;;
    all)           run_detectability; run_robustness ;;
    *) echo "usage: $0 [detectability|robustness|all]" >&2; exit 2 ;;
esac

log "Done. Detectability outputs under $DATA_DIR/phase2_metrics/; robustness tables under $DATA_DIR/phase4_decode/$OUTPUT_SUBDIR/."
