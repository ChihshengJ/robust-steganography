# Reproduction scripts

Thin bash wrappers around the `experiments/phase*` modules. Each one `cd`s to
the repo root, reads a few env-var knobs (with sensible defaults), and forwards
any extra flags straight to the underlying `python -m ...` module — so anything
the module accepts still works:

```bash
SYSTEM=topicqa scripts/phase1_generate.sh --limit 5     # extra flags pass through
```

Run from anywhere; they locate the repo root themselves.

## Files

| Script                 | What it runs                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `serve_local_model.sh` | Launch the pinned llama.cpp server (TopicQA/Story only). Pointer to `experiments/serve_local_model.sh`. |
| `phase1_generate.sh`   | Phase 1 — generate stego / cover texts.                                                                 |
| `phase2_metrics.sh`    | Phase 2 — token counts, perplexity, stegoanalysis classifier + summary.                                 |
| `phase3_attacks.sh`    | Phase 3 — apply the 5 attacks to stegos + covers.                                                       |
| `phase4_decode.sh`     | Phase 4 — decode attacked stegos (BER) + attack-severity metrics.                                       |
| `run_all.sh`           | End-to-end native-capacity run of both data tracks.                                                     |
| `smoke_test.sh`        | Free setup check (imports + dry-runs); validate before `run_all.sh`.                                    |

## Common env knobs

| Var                             | Default            | Meaning                                                                                   |
| ------------------------------- | ------------------ | ----------------------------------------------------------------------------------------- |
| `PYTHON`                        | `uv run python`*   | Interpreter. *Auto-detects uv; falls back to `python` if uv is absent. Override to pin.   |
| `DATA_DIR`                      | `data/experiments` | Base dir for inputs/outputs.                                                              |
| `SYSTEM`                        | `all`              | `topicqa` \| `story` \| `litreview` \| `baseline` \| `all`.                               |
| `CAPACITY`                      | —                  | Message bits; selects the `{system}_cap{N}/` robustness subdir. Requires `SYSTEM != all`. |
| `SUBDIR`                        | module default     | Override the data subdir. Set `SUBDIR=''` for top-level (the detectability dataset).      |
| `MAX_WORKERS`                   | `8`                | Phase 3 attack concurrency.                                                               |
| `LOCAL_BASE_URL`, `LOCAL_MODEL` | see `.env.example` | Local llama.cpp server for TopicQA/Story.                                                 |

## Two data tracks

- **Steganalysis** (Exp 2 & 5): full 300/class at native capacity, written to
  **top-level** `phase1_texts/`, consumed by Phase 2.
  `SUBDIR='' scripts/phase1_generate.sh` → `scripts/phase2_metrics.sh`.
- **Robustness** (Exp 1 & 3): per-system native capacity in `{system}_cap{N}/`,
  first 30 stegos attacked and decoded.
  `CAPACITY=N scripts/phase1_generate.sh` → `phase3_attacks.sh` → `phase4_decode.sh`.

`run_all.sh` does both. See the repo `README.md` for the full walkthrough,
prerequisites, and the phase→experiment map.
