"""Phase 3: Apply attacks to stego (S) and same-pipeline cover (C1) texts.

For each system, take the first 30 stego texts and the first 20 cover-C1 texts from
Phase 1, apply the 5 attack configurations from experiment.md (synonym, local/global
paraphrase, local/global back-translation) to the stegos, and a trimmed cover plan
(local_paraphrase at tampering {0.5, 1.0}, 1 run each) to the covers.

Output layout (matches experiment.md lines 52-55):

    data/experiments/phase3_attacks/
        topicqa_attacked.jsonl
        story_attacked.jsonl
        litreview_attacked.jsonl

Per-system record counts:
    Stego:  30 src x (synonym 3 + local_paraphrase 9 + local_BT 9 + global_paraphrase 3 + global_BT 3)
            = 30 x 27 = 810
    Cover:  20 src x local_paraphrase at {0.5, 1.0} x 1 run = 20 x 2 = 40
    Total per system: 850 attacked records.

Concurrency: API-bound attack calls are dispatched via a ThreadPoolExecutor
(default 8 workers, override with --max-workers). The OpenAI client is
thread-safe; the `random` module is shared global state so the per-task seed
in `derive_seed` is best-effort under concurrency. Set --max-workers 1 to
restore strict deterministic seeding.

Usage:
    python -m experiments.phase3_attacks --system topicqa
    python -m experiments.phase3_attacks --system all --max-workers 16
    python -m experiments.phase3_attacks --system topicqa --n-stegos 2 --skip-covers \
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
from experiments.utils.io import append_jsonl, load_completed_ids, read_jsonl
from experiments.utils.system_factory import make_clients
from experiments.utils.token_counter import count_tokens

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Attack configuration (verbatim from experiment.md lines 233-248)
# ---------------------------------------------------------------------------

ATTACK_CONFIGS: list[dict] = [
    {
        "label": "synonym",
        "attack_type": "synonym",
        "local": True,
        "tampering_levels": [0.2, 0.5, 1.0],
        "runs_per_stego": 1,
    },
    {
        "label": "local_paraphrase",
        "attack_type": "paraphrase",
        "local": True,
        "tampering_levels": [0.2, 0.5, 1.0],
        "runs_per_stego": 3,
    },
    {
        "label": "local_backtranslation",
        "attack_type": "translate",
        "local": True,
        "tampering_levels": [0.2, 0.5, 1.0],
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
        "tampering_levels": [1.0],
        "runs_per_stego": 3,
    },
]

SYSTEMS = ("topicqa", "story", "litreview", "baseline")
# In-house token-level baselines: selectable explicitly but excluded from "all".
BASELINE_LM_SYSTEMS = ("meteor", "discop")

# Cover-attack plan: covers carry no bits, so they only feed Exp 1 attack-severity
# baselines. local_paraphrase at medium and maximum tampering is enough to anchor
# what a paraphrase-class attack does to non-stego text. (Reduced from the full
# 5-attack grid to cut API cost — see experiment.md discussion.)
COVER_ALLOWED: set[tuple[str, float]] = {
    ("local_paraphrase", 0.5),
    ("local_paraphrase", 1.0),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_attacks(client) -> dict[str, object]:
    """Instantiate one attack object per attack_label.

    Translation temp is 0.7 (matches paraphrase) so the 3 runs sample real variance.
    """
    return {
        "synonym": SynonymAttack(method="wordnet"),
        "local_paraphrase": ParaphraseAttack(
            client=client, model="gpt-4.1", temperature=0.7
        ),
        "local_backtranslation": TranslationAttack(
            client=client, model="gpt-4.1", temperature=0.7
        ),
        "global_paraphrase": ParaphraseAttack(
            client=client, model="gpt-4.1", temperature=0.7
        ),
        "global_backtranslation": TranslationAttack(
            client=client, model="gpt-4.1", temperature=0.7
        ),
    }


def build_record_id(
    source_id: str, attack_label: str, tampering: float, run_idx: int
) -> str:
    """Composite id, e.g. topicqa_s_000_global_paraphrase_1.0_run0."""
    return f"{source_id}_{attack_label}_{tampering}_run{run_idx}"


def derive_seed(
    source_id: str, attack_label: str, tampering: float, run_idx: int
) -> int:
    """Deterministic seed in [0, 2**31) for sentence-selection reproducibility."""
    key = f"{source_id}|{attack_label}|{tampering}|{run_idx}"
    return abs(hash(key)) & 0x7FFFFFFF


def load_sources(
    phase1_dir: Path, system: str, n_stegos: int, n_covers: int, skip_covers: bool
) -> list[tuple[dict, str]]:
    """Load (record, text_type) tuples for all sources to attack.

    Stegos: first n_stegos by prompt_idx from {system}_stego.jsonl.
    Covers: first n_covers by prompt_idx from {system}_cover_c1.jsonl (unless skipped).
    """
    sources: list[tuple[dict, str]] = []

    stego_path = phase1_dir / f"{system}_stego.jsonl"
    stego_records = sorted(
        (r for r in read_jsonl(stego_path) if r.get("prompt_idx") is not None),
        key=lambda r: r["prompt_idx"],
    )
    stego_records = [r for r in stego_records if r["prompt_idx"] < n_stegos][:n_stegos]
    sources.extend((r, "stego") for r in stego_records)

    if not skip_covers:
        cover_path = phase1_dir / f"{system}_cover_c1.jsonl"
        cover_records = sorted(
            (r for r in read_jsonl(cover_path) if r.get("prompt_idx") is not None),
            key=lambda r: r["prompt_idx"],
        )
        cover_records = [r for r in cover_records if r["prompt_idx"] < n_covers][
            :n_covers
        ]
        sources.extend((r, "cover_c1") for r in cover_records)

    return sources


def plan_records(
    sources: list[tuple[dict, str]],
    attack_filter: set[str] | None,
) -> list[tuple[dict, str, dict, float, int]]:
    """Build the full (source, source_text_type, attack_cfg, tampering, run_idx) plan.

    Stego sources run the full ATTACK_CONFIGS grid. Cover sources are restricted
    to COVER_ALLOWED (label, tampering) pairs with 1 run each.
    """
    plan = []
    for source, source_text_type in sources:
        for cfg in ATTACK_CONFIGS:
            if attack_filter and cfg["label"] not in attack_filter:
                continue
            for tampering in cfg["tampering_levels"]:
                if source_text_type == "cover_c1":
                    if (cfg["label"], tampering) not in COVER_ALLOWED:
                        continue
                    n_runs = 1
                else:
                    n_runs = cfg["runs_per_stego"]
                for run_idx in range(n_runs):
                    plan.append((source, source_text_type, cfg, tampering, run_idx))
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
) -> dict:
    original_text = source["text"]
    record = {
        "id": build_record_id(source["id"], cfg["label"], tampering, run_idx),
        "source_id": source["id"],
        "source_text_type": source_text_type,
        "system": source["system"],
        "attack_label": cfg["label"],
        "attack_type": cfg["attack_type"],
        "local": cfg["local"],
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
    )
    return record["id"], record, error


def run_system(
    system: str,
    client,
    phase1_dir: Path,
    output_dir: Path,
    n_stegos: int,
    n_covers: int,
    skip_covers: bool,
    attack_filter: set[str] | None,
    dry_run: bool,
    max_workers: int,
):
    out_path = output_dir / f"{system}_attacked.jsonl"
    sources = load_sources(phase1_dir, system, n_stegos, n_covers, skip_covers)
    plan = plan_records(sources, attack_filter)

    n_stego_src = sum(1 for _, t in sources if t == "stego")
    n_cover_src = sum(1 for _, t in sources if t == "cover_c1")
    log.info(
        f"[{system}] sources: {n_stego_src} stegos + {n_cover_src} covers; "
        f"planned records: {len(plan)}; output: {out_path}"
    )

    if dry_run:
        for source, src_type, cfg, tp, run_idx in plan[:3]:
            rid = build_record_id(source["id"], cfg["label"], tp, run_idx)
            log.info(f"  e.g. {rid} ({src_type})")
        if len(plan) > 3:
            log.info(f"  ... and {len(plan) - 3} more")
        return

    attacks = build_attacks(client)
    completed = load_completed_ids(out_path)
    log.info(f"[{system}] {len(completed)} records already done; resuming")

    pending: list[tuple[dict, str, dict, float, int]] = []
    n_skipped = 0
    for task in plan:
        source, _, cfg, tampering, run_idx = task
        rid = build_record_id(source["id"], cfg["label"], tampering, run_idx)
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
        futures = [ex.submit(execute_task, task, attacks) for task in pending]
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
        help="Which system(s) to attack ('all' excludes meteor/discop)",
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
            "auto-resolves --subdir to '{system}_cap{N}' so attacks read the right Phase 1 variant."
        ),
    )
    parser.add_argument(
        "--n-stegos",
        type=int,
        default=30,
        help="Number of stego texts to attack per system (default 30)",
    )
    parser.add_argument(
        "--n-covers",
        type=int,
        default=20,
        help="Number of cover_c1 texts to attack per system (default 20)",
    )
    parser.add_argument(
        "--skip-covers",
        action="store_true",
        help="Skip cover_c1 attacks (only attack stegos)",
    )
    parser.add_argument(
        "--attack",
        action="append",
        choices=[c["label"] for c in ATTACK_CONFIGS],
        default=None,
        help="Filter to one or more attack_labels (repeatable). Default: all 5 attacks.",
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
        "--dry-run",
        action="store_true",
        help="Print planned counts without making API calls",
    )
    args = parser.parse_args()

    if args.capacity is not None:
        if args.system == "all":
            parser.error(
                "--capacity requires --system to be one of topicqa/story/litreview/baseline (not 'all')."
            )
        if args.subdir == "recovery_test":
            args.subdir = f"{args.system}_cap{args.capacity}"
            log.info(f"--capacity set: defaulting --subdir to {args.subdir!r}")

    phase1_dir = args.data_dir / "phase1_texts"
    output_dir = args.data_dir / "phase3_attacks"
    if args.subdir:
        phase1_dir = phase1_dir / args.subdir
        output_dir = output_dir / args.subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Phase 1 inputs: %s", phase1_dir)
    log.info("Phase 3 outputs: %s", output_dir)

    attack_filter = set(args.attack) if args.attack else None

    if args.dry_run:
        client = None
    else:
        client, _local_client = make_clients()

    targets = SYSTEMS if args.system == "all" else (args.system,)
    for system in targets:
        run_system(
            system=system,
            client=client,
            phase1_dir=phase1_dir,
            output_dir=output_dir,
            n_stegos=args.n_stegos,
            n_covers=args.n_covers,
            skip_covers=args.skip_covers,
            attack_filter=attack_filter,
            dry_run=args.dry_run,
            max_workers=max(1, args.max_workers),
        )

    log.info("Phase 3 attacks complete.")


if __name__ == "__main__":
    main()
