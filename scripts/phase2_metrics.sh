#!/usr/bin/env bash
#
# Phase 2 — steganalysis on the detection set: every stegotext of the detection
# cells (phase1_texts/detection/{system}_cap{F}_{config}/, selected inputs only)
# against its length-matched normal generation. Every detector sees the task
# input x next to the text. Generate the cells with TRACK=detection
# scripts/phase1_generate.sh, then scripts/phase1_normal.sh.
#
# Runs, in order:
#   phase2a_token_counts   — token/word counts, length matching per cell   (cpu)
#   phase2b_perplexity     — gpt2-large perplexity conditioned on x         (gpu/mps/cpu)
#   phase2c_transformer    — DistilBERT on (x, text), leave-one-config-out  (gpu recommended)
#   phase2c_summary        — perplexity-only detector + all metrics
#
# Optional extra stegoanalysis signals (off by default, need API keys):
#   RUN_EMBEDDINGS=1  -> phase2c_embeddings (text-embedding-3-large, OPENAI_API_KEY)
#   RUN_LLM_JUDGE=1   -> phase2c_llm_judge  (OPENROUTER_API_KEY)
#   RUN_QUALITY=1     -> phase2f_quality_judge, pairwise text quality of each
#                        stegotext vs its normal generation (TOGETHER_API_KEY)
#
# Env knobs: SYSTEMS (comma list, default all), PPL_MODEL (default gpt2-large),
#            DATA_DIR, PYTHON.
# Extra args are forwarded to phase2c_transformer (the main classifier).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PPL_MODEL="${PPL_MODEL:-gpt2-large}"

common=(--data-dir "$DATA_DIR")
[ -n "${SYSTEMS:-}" ] && common+=(--systems "$SYSTEMS")

run_py experiments.phase2_metrics.phase2a_token_counts "${common[@]}"
run_py experiments.phase2_metrics.phase2b_perplexity "${common[@]}" --model "$PPL_MODEL"
run_py experiments.phase2_metrics.phase2c_transformer "${common[@]}" "$@"

if [ "${RUN_EMBEDDINGS:-0}" = "1" ]; then
    run_py experiments.phase2_metrics.phase2c_embeddings "${common[@]}"
fi
if [ "${RUN_LLM_JUDGE:-0}" = "1" ]; then
    run_py experiments.phase2_metrics.phase2c_llm_judge "${common[@]}"
fi

run_py experiments.phase2_metrics.phase2c_summary "${common[@]}" --ppl-model "$PPL_MODEL"

if [ "${RUN_QUALITY:-0}" = "1" ]; then
    run_py experiments.phase2_metrics.phase2f_quality_judge "${common[@]}"
fi
