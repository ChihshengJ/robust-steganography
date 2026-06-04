"""Phase 1a: Launder existing stego texts through a global paraphrase pass.

Experiment A. The Phase 2b classifier separates stego (S) from prompted cover
(C2) almost perfectly — but Phase 2a shows that signal is a *pipeline*
artifact (S vs C1 is at chance), not a property of the
encoding. This script tests whether a single global-paraphrase pass launders
the pipeline fingerprint away: paraphrasing rewrites S through GPT-4.1, the
same model that produces C2, so paraphrase(S) and C2 should converge.

The global-paraphrase BER measured in Phase 4 (~0.11 raw, recoverable under
RepetitionCode) confirms the hidden message survives this pass — so laundering
costs robustness little while (hypothetically) closing the detectability gap.

This script ONLY generates the laundered texts. The downstream classifier run
(paraphrase(S) vs C2) is a separate step that reuses the Phase 2c machinery.

Input  : data/experiments/phase1_texts/{system}_stego.jsonl   (all 300 stegos)
Output : data/experiments/phase1a_laundered/{system}_stego_laundered.jsonl

Each output record mirrors the Phase 1 schema closely enough to be a drop-in
"stego" input for the stegoanalysis classifiers: `text` holds the laundered
text, `prompt_idx` / `message_bits` / `system_state` are carried over verbatim
so the record can also be re-decoded later if needed.

Attack config matches Phase 3's `global_paraphrase`:
    ParaphraseAttack(model="gpt-4.1", temperature=0.7), called local=False.

Usage:
    python -m experiments.phase1_generation.phase1a_launder --system all
    python -m experiments.phase1_generation.phase1a_launder --system topicqa --n-stegos 5   # smoke test
    python -m experiments.phase1_generation.phase1a_launder --system all --dry-run
    python -m experiments.phase1_generation.phase1a_launder --system story --runs 3 --max-workers 16
"""

from __future__ import annotations

import argparse
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.utils.io import append_jsonl, load_completed_ids, read_jsonl
from experiments.utils.system_factory import make_clients
from experiments.utils.token_counter import count_tokens
from attacks.paraphrase import ParaphraseAttack

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

SYSTEMS = ("topicqa", "story", "litreview")

# Attack parameters — verbatim from Phase 3's `global_paraphrase` config so the
# laundering pass is identical to the attack already validated in Phase 3/4.
PARAPHRASE_MODEL = "gpt-4.1"
PARAPHRASE_TEMPERATURE = 0.7
TAMPERING = 1.0  # global paraphrase ignores the level, but pass it for clarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_record_id(source_id: str, run_idx: int) -> str:
    """Composite id, e.g. topicqa_s_000_laundered_run0."""
    return f"{source_id}_laundered_run{run_idx}"


def derive_seed(source_id: str, run_idx: int) -> int:
    """Deterministic seed in [0, 2**31). Global paraphrase variance comes from
    temperature, not the RNG, so this is purely for record-keeping / parity
    with Phase 3 — and matters only if --runs is combined with local mode."""
    return abs(hash(f"{source_id}|laundered|{run_idx}")) & 0x7FFFFFFF


def load_stegos(stego_path: Path, n_stegos: int | None) -> list[dict]:
    """Load stego records sorted by prompt_idx; optionally cap to first n."""
    records = sorted(
        (r for r in read_jsonl(stego_path) if r.get("prompt_idx") is not None),
        key=lambda r: r["prompt_idx"],
    )
    if n_stegos is not None:
        records = records[:n_stegos]
    return records


def word_count(text: str) -> int:
    return len(text.split())


def make_record(source: dict, run_idx: int, laundered: str | None,
                 error: str | None, seed: int) -> dict:
    """Build one output record. `text` holds the laundered text so the record
    is a drop-in classifier input; `original_text` keeps the Phase 1 stego."""
    original_text = source["text"]
    # ParaphraseAttack._global_paraphrase swallows exceptions and returns the
    # input unchanged; flag that so a silent no-op never pollutes the classifier
    # set. Also flag if the [key points] scaffold leaked into the output.
    suspect = None
    if error is not None:
        suspect = "attack_error"
    elif laundered is not None:
        if laundered.strip() == original_text.strip():
            suspect = "unchanged_text"
        elif "[key points]" in laundered.lower():
            suspect = "marker_leak"

    record = {
        "id": build_record_id(source["id"], run_idx),
        "source_id": source["id"],
        "system": source["system"],
        "text_type": "stego_laundered",
        "prompt_idx": source.get("prompt_idx"),
        "prompt": source.get("prompt"),
        "message_bits": source.get("message_bits"),
        "run_idx": run_idx,
        "original_text": original_text,
        "text": laundered,
        "original_token_count": source.get("token_count") or count_tokens(original_text),
        "token_count": count_tokens(laundered) if laundered else None,
        "word_count": word_count(laundered) if laundered else None,
        "char_count": len(laundered) if laundered else None,
        "system_state": source.get("system_state"),
        "metadata": {
            "attack": "global_paraphrase",
            "tampering_level": TAMPERING,
            "local": False,
            "model": PARAPHRASE_MODEL,
            "temperature": PARAPHRASE_TEMPERATURE,
            "rng_seed": seed,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        record["error"] = error
    if suspect is not None:
        record["suspect"] = suspect
    return record


def execute_task(source: dict, run_idx: int, attack: ParaphraseAttack) -> dict:
    """Launder one stego text. Runs in a worker thread (OpenAI client is
    thread-safe). Returns the finished record."""
    seed = derive_seed(source["id"], run_idx)
    random.seed(seed)
    np.random.seed(seed)
    error = None
    laundered = None
    try:
        laundered = attack(source["text"], TAMPERING, False)  # local=False -> global
    except Exception as e:  # noqa: BLE001 - record and continue
        error = repr(e)
    return make_record(source, run_idx, laundered, error, seed)


# ---------------------------------------------------------------------------
# Per-system driver
# ---------------------------------------------------------------------------


def run_system(system: str, client, phase1_dir: Path, output_dir: Path,
               n_stegos: int | None, runs: int, dry_run: bool,
               max_workers: int) -> None:
    stego_path = phase1_dir / f"{system}_stego.jsonl"
    out_path = output_dir / f"{system}_stego_laundered.jsonl"

    stegos = load_stegos(stego_path, n_stegos)
    if not stegos:
        log.warning("[%s] no stego records found at %s — skipping", system, stego_path)
        return

    # Plan: every (stego, run_idx) pair.
    plan = [(s, r) for s in stegos for r in range(runs)]
    log.info(
        "[%s] %d stegos x %d run(s) = %d laundering tasks; output: %s",
        system, len(stegos), runs, len(plan), out_path,
    )

    if dry_run:
        for source, run_idx in plan[:3]:
            log.info("  e.g. %s", build_record_id(source["id"], run_idx))
        if len(plan) > 3:
            log.info("  ... and %d more", len(plan) - 3)
        return

    completed = load_completed_ids(out_path)
    pending = [
        (s, r) for s, r in plan
        if build_record_id(s["id"], r) not in completed
    ]
    n_skipped = len(plan) - len(pending)
    if not pending:
        log.info("[%s] nothing to do (%d already done).", system, n_skipped)
        return

    log.info(
        "[%s] dispatching %d tasks across %d worker(s); %d already done",
        system, len(pending), max_workers, n_skipped,
    )

    attack = ParaphraseAttack(
        client=client, model=PARAPHRASE_MODEL, temperature=PARAPHRASE_TEMPERATURE
    )
    write_lock = threading.Lock()
    n_done = n_errors = n_suspect = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(execute_task, s, r, attack) for s, r in pending]
        for fut in as_completed(futures):
            try:
                record = fut.result()
            except Exception as e:  # noqa: BLE001
                n_errors += 1
                log.exception("[%s] task crashed: %r", system, e)
                continue

            if record.get("error"):
                n_errors += 1
                log.warning("[%s] %s failed: %s", system, record["id"], record["error"])
            elif record.get("suspect"):
                n_suspect += 1
                log.warning(
                    "[%s] %s suspect (%s)", system, record["id"], record["suspect"]
                )

            with write_lock:
                append_jsonl(out_path, record)
            n_done += 1
            if n_done % 25 == 0:
                log.info(
                    "[%s] progress: %d/%d new (%d errors, %d suspect)",
                    system, n_done, len(pending), n_errors, n_suspect,
                )

    log.info(
        "[%s] done. wrote %d new records (%d skipped, %d errors, %d suspect)",
        system, n_done, n_skipped, n_errors, n_suspect,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1a: launder stego texts via a global paraphrase pass"
    )
    parser.add_argument(
        "--system", choices=[*SYSTEMS, "all"], default="all",
        help="Which system(s) to launder",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/experiments"),
        help="Base directory for Phase 1 inputs and Phase 1a outputs",
    )
    parser.add_argument(
        "--input-subdir", default="",
        help="Sub-directory under phase1_texts/ to read stegos from "
             "(default: '' = top-level, the full 300-stego files)",
    )
    parser.add_argument(
        "--output-subdir", default="",
        help="Sub-directory under phase1a_laundered/ to write outputs to "
             "(default: '' = top-level)",
    )
    parser.add_argument(
        "--n-stegos", type=int, default=None,
        help="Cap to the first N stegos by prompt_idx (default: all, ~300)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Laundering passes per stego (default 1; >1 samples paraphrase "
             "variance, at temperature 0.7)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Number of concurrent paraphrase workers (default 8)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned counts without making API calls",
    )
    args = parser.parse_args()

    phase1_dir = args.data_dir / "phase1_texts"
    output_dir = args.data_dir / "phase1a_laundered"
    if args.input_subdir:
        phase1_dir = phase1_dir / args.input_subdir
    if args.output_subdir:
        output_dir = output_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Phase 1 stego inputs: %s", phase1_dir)
    log.info("Phase 1a laundered outputs: %s", output_dir)

    client = None if args.dry_run else make_clients()[0]

    targets = SYSTEMS if args.system == "all" else (args.system,)
    for system in targets:
        run_system(
            system=system,
            client=client,
            phase1_dir=phase1_dir,
            output_dir=output_dir,
            n_stegos=args.n_stegos,
            runs=max(1, args.runs),
            dry_run=args.dry_run,
            max_workers=max(1, args.max_workers),
        )

    log.info("Phase 1a laundering complete.")


if __name__ == "__main__":
    main()
