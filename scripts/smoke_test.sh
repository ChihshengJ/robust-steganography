#!/usr/bin/env bash
#
# Fast, (almost) free setup check. Verifies the package imports, every phase
# module loads, and the no-API dry-runs plan correctly — so you catch missing
# deps / env problems before launching the real, expensive run_all.sh.
#
# Spends NO OpenAI credit by default. Set RUN_API_SMOKE=1 to also generate one
# tiny LitReview stego (1 prompt, no local server needed) to validate the API
# path end to end — that costs a couple of OpenAI calls.
#
# Env knobs: DATA_DIR, PYTHON, RUN_API_SMOKE.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

log "1/3  Import-checking phase modules (free)…"
run "${PYTHON[@]}" - <<'PY'
import importlib
mods = [
    "experiments.phase1_generation.phase1_generate",
    "experiments.phase1_generation.phase1a_launder",
    "experiments.phase2_metrics.phase2a_token_counts",
    "experiments.phase2_metrics.phase2b_perplexity",
    "experiments.phase2_metrics.phase2c_transformer",
    "experiments.phase2_metrics.phase2c_summary",
    "experiments.phase3_attacks",
    "experiments.phase4_decode.phase4a_decode",
    "experiments.phase4_decode.phase4b_attack_metrics",
    "experiments.phase4_decode.phase4c_main_results",
]
for m in mods:
    importlib.import_module(m)
    print("  ok:", m)
PY

log "2/3  Phase 3 dry-run (plans attack counts, no API)…"
run_py experiments.phase3_attacks --data-dir "$DATA_DIR" --dry-run \
    || warn "phase3 dry-run failed (Phase 1 outputs may not exist yet) — non-fatal."

log "3/3  Phase 4a dry-run (plans decode counts, no API)…"
run_py experiments.phase4_decode.phase4a_decode --data-dir "$DATA_DIR" --dry-run \
    || warn "phase4a dry-run failed (Phase 3 outputs may not exist yet) — non-fatal."

if [ "${RUN_API_SMOKE:-0}" = "1" ]; then
    log "API smoke: one LitReview stego (1 prompt, spends a little OpenAI credit)…"
    SYSTEM=litreview SUBDIR='' "$REPO_ROOT/scripts/phase1_generate.sh" --limit 1 --stego-only
fi

log "Smoke test complete."
