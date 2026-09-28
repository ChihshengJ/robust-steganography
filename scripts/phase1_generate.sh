#!/usr/bin/env bash
#
# Phase 1 — generate stegotexts. API-heavy. The normal generations steganalysis
# compares them with come from scripts/phase1_normal.sh. Long-form QA and StoryGen additionally need the local llama.cpp
# server up (scripts/serve_local_model.sh); LitReview does not.
#
# Two layouts, selected by how you call it:
#
#   * Steganalysis dataset (full 300/class at native capacity, top-level
#     phase1_texts/ — this is what Phase 2 reads):
#         SYSTEM=story SUBDIR='' scripts/phase1_generate.sh
#
#   * Robustness dataset (per-system native capacity in {system}_cap{N}/ —
#     what Phase 3/4 read). Pass CAPACITY; the module auto-names the subdir:
#         SYSTEM=story   CAPACITY=16 scripts/phase1_generate.sh   # 16 slots (G asked for 18)
#         SYSTEM=litreview CAPACITY=20 scripts/phase1_generate.sh
#
#   * Payload-grid cells (ARR revision): TRACK=detection|recovery writes the
#     cell under phase1_texts/{track}/, since detection and recovery texts are
#     generated separately. LitReview's grid inputs are LITREVIEW_INDICES=
#     litreview_indices_min80.json. Generate ~40 inputs (--limit 40), then pick
#     the shared 30 with experiments.phase1_generation.select_inputs:
#         SYSTEM=litreview CAPACITY=16 TRACK=recovery SYNTH_PROVIDER=openrouter \
#             SYNTH_MODEL=deepseek/deepseek-v4.1-flash \
#             LITREVIEW_INDICES=litreview_indices_min80.json \
#             scripts/phase1_generate.sh --limit 40
#
# Configuration (story/litreview; unset = the default configuration):
#     SYNTH_MODEL, SYNTH_PROVIDER,
#     GENERATOR_MODEL, GENERATOR_PROVIDER, GENERATOR_EXTRA_BODY (story only).
# A non-default configuration writes to a tagged subdir:
#     SYSTEM=litreview CAPACITY=16 SYNTH_PROVIDER=openrouter \
#         SYNTH_MODEL=deepseek/deepseek-v4.1-flash scripts/phase1_generate.sh
#
# Env knobs: SYSTEM (default all), CAPACITY, TRACK, SUBDIR, LITREVIEW_INDICES,
# DATA_DIR, PYTHON.
# Anything else is forwarded, e.g.:  scripts/phase1_generate.sh --limit 5 --dry-run
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SYSTEM="${SYSTEM:-all}"

args=(--system "$SYSTEM" --data-dir "$DATA_DIR")
[ -n "${CAPACITY:-}" ] && args+=(--capacity "$CAPACITY")
[ -n "${TRACK:-}" ] && args+=(--track "$TRACK")
[ -n "${LITREVIEW_INDICES:-}" ] && args+=(--litreview-indices "$LITREVIEW_INDICES")
# SUBDIR is honoured even when set to the empty string (selects top-level).
[ -n "${SUBDIR+x}" ] && args+=(--subdir "$SUBDIR")
[ -n "${SYNTH_MODEL:-}" ] && args+=(--synth-model "$SYNTH_MODEL")
[ -n "${SYNTH_PROVIDER:-}" ] && args+=(--synth-provider "$SYNTH_PROVIDER")
[ -n "${GENERATOR_MODEL:-}" ] && args+=(--generator-model "$GENERATOR_MODEL")
[ -n "${GENERATOR_PROVIDER:-}" ] && args+=(--generator-provider "$GENERATOR_PROVIDER")
[ -n "${GENERATOR_EXTRA_BODY:-}" ] && args+=(--generator-extra-body "$GENERATOR_EXTRA_BODY")

run_py experiments.phase1_generation.phase1_generate "${args[@]}" "$@"
