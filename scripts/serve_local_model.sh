#!/usr/bin/env bash
#
# Thin pointer to the canonical launcher at experiments/serve_local_model.sh,
# so all reproduction entrypoints live under scripts/. Launches llama-server
# pinned for DETERMINISTIC decoding (needed by TopicQA / Story generation AND
# decoding). Run it in its own terminal; it stays in the foreground.
#
#   scripts/serve_local_model.sh /path/to/Qwen3.5-4B-UD-Q8_K_XL.gguf
#   PORT=8080 NGL=0 scripts/serve_local_model.sh /path/to/model.gguf
#
# The model basename you serve must match $LOCAL_MODEL (see .env.example).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../experiments/serve_local_model.sh" "$@"
