#!/usr/bin/env bash
#
# serve_local_model.sh -- launch llama-server pinned for DETERMINISTIC decoding.
#
# TopicQA and Story decode by RE-generating subtopics/slots with the local
# model. That only reproduces the encode-time output if every call sees the
# identical computation. This script pins the serve command so Phase 1
# (generation) and Phase 4 (decoding) are byte-for-byte the same.
#
#   >>> USE THIS EXACT SCRIPT FOR BOTH GENERATION AND DECODING. <<<
#   Do not edit the flags between phases. Do not update llama.cpp mid-run.
#   If you must override a knob, pass the SAME env vars both times.
#
# What is pinned and why:
#   --parallel 1          single slot: no concurrent requests => no cross-
#                         request batching (the #1 cause of run-to-run drift).
#   --no-cont-batching    continuous batching off (belt-and-suspenders with -np 1).
#   --batch-size/--ubatch-size 2048
#                         prompt processed in ONE prefill pass (subtopic/slot
#                         prompts are far shorter than 2048 tokens) => no
#                         chunked-prefill reduction-order variance.
#   -ngl 0                pure CPU by default: no GPU-kernel / atomic-reduction
#                         non-determinism. Override with NGL=... if you need
#                         speed, but then you MUST use the same NGL every time.
#   -t (fixed)            fixed thread count => fixed reduction order.
#   --temp 0 / --seed 0   greedy, fixed seed.
#   --jinja               required: generate_subtopics() sends
#                         chat_template_kwargs={"enable_thinking": false},
#                         which llama-server only honors with --jinja.
#
# Usage:
#   ./experiments/serve_local_model.sh /path/to/Qwen3.5-4B-UD-Q8_K_XL.gguf
#   PORT=8080 NGL=0 THREADS=8 ./experiments/serve_local_model.sh /path/to/model.gguf
#
# Runs in the foreground -- launch it in its own terminal; Ctrl-C to stop.

set -euo pipefail

# ---------------------------------------------------------------------------
# Input: path to the GGUF model file.
# ---------------------------------------------------------------------------
MODEL_PATH="${1:?Usage: $0 /path/to/model.gguf}"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "error: model file not found: $MODEL_PATH" >&2
    exit 1
fi

if ! command -v llama-server >/dev/null 2>&1; then
    echo "error: 'llama-server' not on PATH (install/build llama.cpp first)." >&2
    exit 1
fi

# Parse the model name from the path: the filename, used as the server alias
# (the name reported on /v1/models and matched against API 'model' fields).
MODEL_FILE="$(basename "$MODEL_PATH")"

# ---------------------------------------------------------------------------
# Knobs -- override via env vars. Whatever you choose, use the SAME values for
# generation and decoding; consistency is what guarantees reproduction.
# ---------------------------------------------------------------------------
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}" # must match LOCAL_BASE_URL in experiments/utils/system_factory.py
CTX="${CTX:-1024}"
NGL="${NGL:-0}" # GPU layers offloaded. 0 = pure CPU (most deterministic).

if [[ -z "${THREADS:-}" ]]; then
    if command -v sysctl >/dev/null 2>&1 && sysctl -n hw.physicalcpu >/dev/null 2>&1; then
        THREADS="$(sysctl -n hw.physicalcpu)" # macOS
    elif command -v nproc >/dev/null 2>&1; then
        THREADS="$(nproc)" # Linux
    else
        THREADS=8
    fi
fi

# ---------------------------------------------------------------------------
# Build the pinned command.
# ---------------------------------------------------------------------------
ARGS=(
    -m "$MODEL_PATH"
    -a "$MODEL_FILE"
    --host "$HOST"
    --port "$PORT"
    -c "$CTX"
    -ngl "$NGL"
    -t "$THREADS"
    --parallel 1
    --no-cont-batching
    --batch-size 2048
    --ubatch-size 2048
    --temp 0
    --seed 0
    --jinja
    --no-warmup
    --mlock
)

echo "=============================================================="
echo " serve_local_model.sh -- deterministic llama-server"
echo "=============================================================="
echo " model path : $MODEL_PATH"
echo " model name : $MODEL_FILE   (server alias)"
echo " endpoint   : http://$HOST:$PORT/v1"
echo " ctx        : $CTX"
echo " ngl        : $NGL    (0 = pure CPU)"
echo " threads    : $THREADS"
echo " batching   : OFF (--parallel 1 --no-cont-batching, single-pass prefill)"
echo "--------------------------------------------------------------"
echo " Use this SAME script + SAME model for Phase 1 and Phase 4."
echo " Gate before spending on attacks:"
echo "   python -m experiments.check_subtopic_repro --base-url http://$HOST:$PORT/v1"
echo "=============================================================="
echo "+ llama-server ${ARGS[*]}"
echo

exec llama-server "${ARGS[@]}"
