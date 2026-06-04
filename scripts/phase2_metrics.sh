#!/usr/bin/env bash
#
# Phase 2 — local metrics + steganalysis (Experiments 2 & 5). No OpenAI calls
# for the default signals; reads the top-level phase1_texts/ detectability
# dataset (generate it with `SUBDIR='' scripts/phase1_generate.sh`).
#
# Runs, in order:
#   phase2a_token_counts   — token/word counts, bits/token            (cpu)
#   phase2b_perplexity     — gpt2-large perplexity                    (gpu/mps/cpu)
#   phase2c_transformer    — DistilBERT 5-fold CV detector            (gpu recommended)
#   phase2c_summary        — aggregate all stegoanalysis signals
#
# Optional extra stegoanalysis signals (off by default, need extra setup):
#   RUN_EMBEDDINGS=1  -> phase2c_embeddings (Qwen3 embed server + GOOGLE_API_KEY)
#   RUN_LLM_JUDGE=1   -> phase2c_llm_judge  (OPENROUTER_API_KEY)
#
# Env knobs: SYSTEMS (comma list, default all), SUB_EXPERIMENT (2a|2b|both),
#            PPL_MODEL (default gpt2-large), DATA_DIR, PYTHON.
# Extra args are forwarded to phase2c_transformer (the main classifier).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SUB_EXPERIMENT="${SUB_EXPERIMENT:-both}"
PPL_MODEL="${PPL_MODEL:-gpt2-large}"

common=(--data-dir "$DATA_DIR")
[ -n "${SYSTEMS:-}" ] && common+=(--systems "$SYSTEMS")

run_py experiments.phase2_metrics.phase2a_token_counts "${common[@]}"
run_py experiments.phase2_metrics.phase2b_perplexity "${common[@]}" --model "$PPL_MODEL"

stego=("${common[@]}" --sub-experiment "$SUB_EXPERIMENT")
run_py experiments.phase2_metrics.phase2c_transformer "${stego[@]}" "$@"

if [ "${RUN_EMBEDDINGS:-0}" = "1" ]; then
    run_py experiments.phase2_metrics.phase2c_embeddings "${stego[@]}"
fi
if [ "${RUN_LLM_JUDGE:-0}" = "1" ]; then
    run_py experiments.phase2_metrics.phase2c_llm_judge "${stego[@]}"
fi

run_py experiments.phase2_metrics.phase2c_summary "${stego[@]}"
