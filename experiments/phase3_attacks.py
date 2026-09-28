"""Phase 3: Apply attacks to stegotexts for recovery evaluation.

For each system, take the first 30 stegotexts of a Phase 1 cell (its selected
inputs, see select_inputs) and apply the attack set (ATTACK_CONFIGS): synonym,
local paraphrase at p in {0.5, 1.0}, global paraphrase, and global round-trip
translation through Japanese.

Output layout:

    data/experiments/phase3_attacks/{subdir}/{system}_attacked.jsonl

LLM attacks run on --attacker-model (default zai-org/GLM-5.3-Flash on Together;
--attacker-provider openai for OpenAI models). Each record
stores it as `attacker_model` (None for synonym), and records from an attacker
other than the camera-ready gpt-4.1 get an `_atk-{slug}` id suffix, so several
attackers can share one attacked file. Sentence-selection seeds don't depend on
the attacker, so local attacks by different models touch the same sentences of
a given text.

An attack that fails after its retries (API errors, unusable output) is
written to {system}_attack_failures.jsonl instead, never to the attacked file,
so Phase 4 cannot decode it as if it were an attacked text. Rerunning the same
command retries every record that is not yet in the attacked file.

Per-cell record count (30 stegotexts):
    30 x (synonym 3 + local_paraphrase 2 x 3 + global_paraphrase 3
          + global_backtranslation 3) = 30 x 15 = 450

Concurrency: API-bound attack calls are dispatched via a ThreadPoolExecutor
(default 8 workers, override with --max-workers). The OpenAI client is
thread-safe; the `random` module is shared global state so the per-task seed
in `derive_seed` is best-effort under concurrency. Set --max-workers 1 to
restore strict deterministic seeding.

Usage:
    python -m experiments.phase3_attacks --system story --capacity 16 --track recovery
    python -m experiments.phase3_attacks --system story --n-stegos 2 \
        --attack global_paraphrase            # smoke test
    python -m experiments.phase3_attacks --system all --dry-run
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

from attacks.paraphrase import ParaphraseAttack
from attacks.synonym import SynonymAttack
from attacks.translation import TranslationAttack
from experiments.utils.attackers import (
    ATTACKER_MODEL,
    ATTACKER_PROVIDER,
    DEFAULT_ATTACKER,
    LLM_ATTACK_TYPES,
)
from experiments.utils.io import (
    TRACKS,
    append_jsonl,
    model_slug,
    read_jsonl,
    read_stego_records,
    resolve_subdir,
    stable_seed,
)
from experiments.utils.system_factory import make_clients, make_together_client
from experiments.utils.token_counter import count_tokens

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Attack configuration (ARR October revision, REVISION_PLAN_ARR_OCT.md §1)
# ---------------------------------------------------------------------------

ATTACK_CONFIGS: list[dict] = [
    # Synonym levels stay as in the camera-ready until the context-aware
    # attack replaces the token-level one (p will then mean "fraction of
    # eligible words").
    {
        "label": "synonym",
        "attack_type": "synonym",
        "local": True,
        "tampering_levels": [0.2, 0.5, 1.0],
        "runs_per_stego": 1,
    },
    # local=True is passed to the attack, so p = 1.0 stays sentence by sentence
    # rather than falling back to a global paraphrase.
    {
        "label": "local_paraphrase",
        "attack_type": "paraphrase",
        "local": True,
        "tampering_levels": [0.5, 1.0],
        "runs_per_stego": 3,
    },
    {
        "label": "global_paraphrase",
        "attack_type": "paraphrase",
        "local": False,
        "tampering_levels": [1.0],
        "runs_per_stego": 3,
    },
    {
        "label": "global_backtranslation",
        "attack_type": "translate",
        "local": False,
        "language": "Japanese",
        "tampering_levels": [1.0],
        "runs_per_stego": 3,
    },
]

SYSTEMS = ("story", "litreview")
# In-house token-level baselines: selectable explicitly but excluded from "all".
BASELINE_LM_SYSTEMS = ("discop",)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_attacks(client, model: str = ATTACKER_MODEL) -> dict[str, object]:
    """Instantiate one attack object per attack_label; LLM attacks run on `model`.

    Translation temp is 0.7 (matches paraphrase) so the 3 runs sample real variance.
    """
    pivot = {c["label"]: c.get("language") for c in ATTACK_CONFIGS}
    return {
        "synonym": SynonymAttack(method="wordnet"),
        "local_paraphrase": ParaphraseAttack(
            client=client, model=model, temperature=0.7
        ),
        "global_paraphrase": ParaphraseAttack(
            client=client, model=model, temperature=0.7
        ),
        "global_backtranslation": TranslationAttack(
            client=client,
            model=model,
            temperature=0.7,
            language=pivot["global_backtranslation"],
        ),
    }


def task_attacker(cfg: dict, attacker_model: str) -> str | None:
    """The attacker model a task runs on, or None for an attack with no LLM."""
    return attacker_model if cfg["attack_type"] in LLM_ATTACK_TYPES else None


def build_record_id(
    source_id: str,
    attack_label: str,
    tampering: float,
    run_idx: int,
    attacker: str | None = None,
) -> str:
    """Composite id, e.g. story_s_000_global_paraphrase_1.0_run0.

    A non-default attacker adds a suffix (..._run0_atk-deepseek-v4-flash); the
    default and non-LLM attacks keep the original ids, so existing files resume.
    """
    rid = f"{source_id}_{attack_label}_{tampering}_run{run_idx}"
    if attacker is not None and attacker != DEFAULT_ATTACKER:
        rid += f"_atk-{model_slug(attacker)}"
    return rid


def derive_seed(
    source_id: str, attack_label: str, tampering: float, run_idx: int
) -> int:
    """Deterministic seed in [0, 2**31) for sentence-selection reproducibility."""
    return stable_seed(f"{source_id}|{attack_label}|{tampering}|{run_idx}")


def load_sources(phase1_dir: Path, system: str, n_stegos: int) -> list[dict]:
    """The stegotexts to attack: the first n_stegos *available* by prompt_idx
    from {system}_stego.jsonl, among the selected inputs when the dir has them
    (read_stego_records).

    "Available", not "prompt_idx < n": Phase 1 can legitimately produce no record
    for a prompt — the Discop baseline skips one whose generation degenerated
    (see phase1_generate) — and filtering by index would then silently return
    n-minus-the-gaps sources, quietly shrinking the sample. Taking the first n
    that exist is identical when there are no gaps.
    """
    stego_records = sorted(
        (
            r
            for r in read_stego_records(phase1_dir, system)
            if r.get("prompt_idx") is not None
        ),
        key=lambda r: r["prompt_idx"],
    )[:n_stegos]
    if len(stego_records) < n_stegos:
        log.warning(
            "[%s] only %d stego records available, wanted %d — generate more "
            "prompts in Phase 1 if the sample matters.",
            system,
            len(stego_records),
            n_stegos,
        )
    return stego_records


def plan_records(
    sources: list[dict],
    attack_filter: set[str] | None,
) -> list[tuple[dict, str, dict, float, int]]:
    """Build the full (source, source_text_type, attack_cfg, tampering, run_idx)
    plan: every stegotext runs the full ATTACK_CONFIGS grid."""
    plan = []
    for source in sources:
        for cfg in ATTACK_CONFIGS:
            if attack_filter and cfg["label"] not in attack_filter:
                continue
            for tampering in cfg["tampering_levels"]:
                for run_idx in range(cfg["runs_per_stego"]):
                    plan.append((source, "stego", cfg, tampering, run_idx))
    return plan


def attack_one(
    attacks: dict,
    cfg: dict,
    text: str,
    tampering: float,
    seed: int,
) -> tuple[str | None, str | None]:
    """Run a single attack call. Returns (attacked_text, error_message_or_None)."""
    random.seed(seed)
    np.random.seed(seed)
    attack = attacks[cfg["label"]]
    try:
        attacked = attack(text, tampering, cfg["local"])
        return attacked, None
    except Exception as e:
        return None, repr(e)


def make_record(
    source: dict,
    source_text_type: str,
    cfg: dict,
    tampering: float,
    run_idx: int,
    attacked_text: str | None,
    error: str | None,
    seed: int,
    attacker: str | None,
) -> dict:
    original_text = source["text"]
    record = {
        "id": build_record_id(source["id"], cfg["label"], tampering, run_idx, attacker),
        "source_id": source["id"],
        "source_text_type": source_text_type,
        "system": source["system"],
        "attack_label": cfg["label"],
        "attack_type": cfg["attack_type"],
        "attacker_model": attacker,
        "local": cfg["local"],
        "pivot_language": cfg.get("language"),
        "tampering_level": tampering,
        "run_idx": run_idx,
        "original_text": original_text,
        "attacked_text": attacked_text,
        "original_token_count": source.get("token_count")
        or count_tokens(original_text),
        "attacked_token_count": count_tokens(attacked_text) if attacked_text else None,
        "system_state": source.get("system_state"),
        "metadata": {"rng_seed": seed},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        record["error"] = error
    return record


# ---------------------------------------------------------------------------
# Main attack loop per system
# ---------------------------------------------------------------------------


def execute_task(
    task: tuple[dict, str, dict, float, int],
    attacks: dict,
    attacker_model: str,
) -> tuple[str, dict, str | None]:
    """Run a single attack task and return (record_id, record, error_or_None).

    Designed to be called from a worker thread. No locking needed inside —
    the OpenAI client is thread-safe, and we let the per-task `random.seed`
    in `attack_one` be best-effort under concurrency (results are cached in
    JSONL, so non-determinism here only affects fresh first runs).
    """
    source, source_text_type, cfg, tampering, run_idx = task
    seed = derive_seed(source["id"], cfg["label"], tampering, run_idx)
    attacked_text, error = attack_one(attacks, cfg, source["text"], tampering, seed)
    record = make_record(
        source=source,
        source_text_type=source_text_type,
        cfg=cfg,
        tampering=tampering,
        run_idx=run_idx,
        attacked_text=attacked_text,
        error=error,
        seed=seed,
        attacker=task_attacker(cfg, attacker_model),
    )
    return record["id"], record, error


def run_system(
    system: str,
    client,
    phase1_dir: Path,
    output_dir: Path,
    n_stegos: int,
    attack_filter: set[str] | None,
    dry_run: bool,
    max_workers: int,
    attacker_model: str = ATTACKER_MODEL,
):
    out_path = output_dir / f"{system}_attacked.jsonl"
    failures_path = output_dir / f"{system}_attack_failures.jsonl"
    sources = load_sources(phase1_dir, system, n_stegos)
    plan = plan_records(sources, attack_filter)

    log.info(
        f"[{system}] sources: {len(sources)} stegos; "
        f"planned records: {len(plan)}; output: {out_path}"
    )

    if dry_run:
        for source, src_type, cfg, tp, run_idx in plan[:3]:
            rid = build_record_id(
                source["id"],
                cfg["label"],
                tp,
                run_idx,
                task_attacker(cfg, attacker_model),
            )
            log.info(f"  e.g. {rid} ({src_type})")
        if len(plan) > 3:
            log.info(f"  ... and {len(plan) - 3} more")
        return

    attacks = build_attacks(client, attacker_model)
    # Files written before failures were split out hold failed records with
    # attacked_text=None; they don't count as done, so a rerun retries them.
    completed = {
        r["id"]
        for r in read_jsonl(out_path)
        if r.get("attacked_text") and not r.get("error")
    }
    log.info(f"[{system}] {len(completed)} records already done; resuming")

    pending: list[tuple[dict, str, dict, float, int]] = []
    n_skipped = 0
    for task in plan:
        source, _, cfg, tampering, run_idx = task
        rid = build_record_id(
            source["id"],
            cfg["label"],
            tampering,
            run_idx,
            task_attacker(cfg, attacker_model),
        )
        if rid in completed:
            n_skipped += 1
            continue
        pending.append(task)

    if not pending:
        log.info(f"[{system}] nothing to attack ({n_skipped} already done).")
        return

    log.info(
        f"[{system}] dispatching {len(pending)} tasks across {max_workers} worker(s); "
        f"{n_skipped} skipped"
    )

    write_lock = threading.Lock()
    n_done_now = 0
    n_errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(execute_task, task, attacks, attacker_model) for task in pending
        ]
        for fut in as_completed(futures):
            try:
                rid, record, error = fut.result()
            except Exception as e:
                n_errors += 1
                log.exception(f"[{system}] task crashed: {e!r}")
                continue

            if error:
                n_errors += 1
                log.warning(f"[{system}] {rid} attack failed: {error}")
                with write_lock:
                    append_jsonl(failures_path, record)
                continue

            with write_lock:
                append_jsonl(out_path, record)
            n_done_now += 1

            if n_done_now % 25 == 0:
                log.info(
                    f"[{system}] progress: {n_done_now}/{len(pending)} new "
                    f"({n_skipped} skipped, {n_errors} errors)"
                )

    log.info(
        f"[{system}] done. wrote {n_done_now} new records "
        f"({n_skipped} skipped, {n_errors} errors)"
    )
    if n_errors:
        log.warning(
            f"[{system}] {n_errors} attacks failed (see {failures_path}); "
            f"rerun the same command to retry them."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Apply attacks to Phase 1 texts"
    )
    parser.add_argument(
        "--system",
        choices=[*SYSTEMS, *BASELINE_LM_SYSTEMS, "all"],
        default="all",
        help="Which system(s) to attack ('all' excludes discop)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/experiments"),
        help="Base directory for Phase 1 inputs and Phase 3 outputs",
    )
    parser.add_argument(
        "--subdir",
        default="recovery_test",
        help=(
            "Sub-directory under phase1_texts/ and phase3_attacks/ to read "
            "inputs from and write outputs to (default: recovery_test). "
            "Pass --subdir '' to use the top-level dirs. "
            "If --capacity is set and --subdir is left at the default, "
            "subdir auto-becomes '{system}_cap{N}'."
        ),
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help=(
            "Convenience flag: when set with --system != all and --subdir at default, "
            "auto-resolves --subdir to '{system}_cap{N}' so attacks read the right "
            "Phase 1 variant (with --track: the track's one cell at that F)."
        ),
    )
    parser.add_argument(
        "--track",
        choices=TRACKS,
        default=None,
        help="Read and write the cell under {track}/ (see phase1_generate --track).",
    )
    parser.add_argument(
        "--n-stegos",
        type=int,
        default=30,
        help="Number of stego texts to attack per system (default 30)",
    )
    parser.add_argument(
        "--attack",
        action="append",
        choices=[c["label"] for c in ATTACK_CONFIGS],
        default=None,
        help="Filter to one or more attack_labels (repeatable). Default: all attacks.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help=(
            "Number of concurrent attack workers (default 12). "
            "Set to 1 for strict deterministic seeding."
        ),
    )
    parser.add_argument(
        "--attacker-model",
        default=ATTACKER_MODEL,
        help=(
            f"Model for the LLM attacks (paraphrase, back-translation); default "
            f"{ATTACKER_MODEL}. Synonym uses no LLM and is unaffected."
        ),
    )
    parser.add_argument(
        "--attacker-provider",
        choices=("openai", "together"),
        default=ATTACKER_PROVIDER,
        help=(
            f"API serving --attacker-model (default {ATTACKER_PROVIDER}; together "
            "needs TOGETHER_API_KEY)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned counts without making API calls",
    )
    args = parser.parse_args()

    if args.capacity is not None and args.system == "all":
        parser.error(
            "--capacity requires --system to be one of story/litreview/discop (not 'all')."
        )
    args.subdir = resolve_subdir(
        args.data_dir / "phase1_texts",
        args.system,
        args.capacity,
        args.subdir,
        args.track,
        default_subdir="recovery_test",
    )

    phase1_dir = args.data_dir / "phase1_texts"
    output_dir = args.data_dir / "phase3_attacks"
    if args.subdir:
        phase1_dir = phase1_dir / args.subdir
        output_dir = output_dir / args.subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Phase 1 inputs: %s", phase1_dir)
    log.info("Phase 3 outputs: %s", output_dir)

    attack_filter = set(args.attack) if args.attack else None

    log.info("Attacker: %s via %s", args.attacker_model, args.attacker_provider)
    if args.dry_run:
        client = None
    elif args.attacker_provider == "together":
        client = make_together_client()
    else:
        client, _generator_client = make_clients()

    targets = SYSTEMS if args.system == "all" else (args.system,)
    for system in targets:
        run_system(
            system=system,
            client=client,
            phase1_dir=phase1_dir,
            output_dir=output_dir,
            n_stegos=args.n_stegos,
            attack_filter=attack_filter,
            dry_run=args.dry_run,
            max_workers=max(1, args.max_workers),
            attacker_model=args.attacker_model,
        )

    log.info("Phase 3 attacks complete.")


if __name__ == "__main__":
    main()
