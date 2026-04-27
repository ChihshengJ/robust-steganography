"""Phase 3: Apply attacks to stego (S) and same-pipeline cover (C1) texts.

For each system, take the first 30 stego texts and the first 20 cover-C1 texts from
Phase 1, apply the 5 attack configurations from experiment.md (synonym, local/global
paraphrase, local/global back-translation), and write attacked texts as JSONL.

Output layout (matches experiment.md lines 52-55):

    data/experiments/phase3_attacks/
        topicqa_attacked.jsonl
        story_attacked.jsonl
        litreview_attacked.jsonl

Per-system record counts:
    Stego:  30 src x (synonym 3 + local_paraphrase 9 + local_BT 9 + global_paraphrase 3 + global_BT 3)
            = 30 x 27 = 810
    Cover:  20 src x (synonym 3 + local_paraphrase 3 + local_BT 3 + global_paraphrase 1 + global_BT 1)
            = 20 x 11 = 220
    Total per system: 1030 attacked records.

Usage:
    python -m experiments.phase3_attacks --system topicqa
    python -m experiments.phase3_attacks --system all
    python -m experiments.phase3_attacks --system topicqa --n-stegos 2 --skip-covers \
        --attack global_paraphrase            # smoke test
    python -m experiments.phase3_attacks --system all --dry-run
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.utils.io import append_jsonl, load_completed_ids, read_jsonl
from experiments.utils.system_factory import make_clients
from experiments.utils.token_counter import count_tokens
from watermarks.attacks.paraphrase import ParaphraseAttack
from watermarks.attacks.synonym import SynonymAttack
from watermarks.attacks.translation import TranslationAttack

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

SYSTEMS = ("topicqa", "story", "litreview")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_attacks(client) -> dict[str, object]:
    """Instantiate one attack object per attack_label.

    Translation temp is 0.7 (matches paraphrase) so the 3 runs sample real variance.
    """
    return {
        "synonym": SynonymAttack(method="wordnet"),
        "local_paraphrase": ParaphraseAttack(client=client, model="gpt-4.1", temperature=0.7),
        "local_backtranslation": TranslationAttack(client=client, model="gpt-4.1", temperature=0.7),
        "global_paraphrase": ParaphraseAttack(client=client, model="gpt-4.1", temperature=0.7),
        "global_backtranslation": TranslationAttack(client=client, model="gpt-4.1", temperature=0.7),
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
        cover_records = [r for r in cover_records if r["prompt_idx"] < n_covers][:n_covers]
        sources.extend((r, "cover_c1") for r in cover_records)

    return sources


def cover_runs(label: str, default_runs: int) -> int:
    """Cover attacks always use 1 run per attack config (experiment.md line 259)."""
    return 1


def plan_records(
    sources: list[tuple[dict, str]],
    attack_filter: set[str] | None,
) -> list[tuple[dict, str, dict, float, int]]:
    """Build the full (source, source_text_type, attack_cfg, tampering, run_idx) plan.

    Used by --dry-run and as the iteration spine of the main loop.
    """
    plan = []
    for source, source_text_type in sources:
        for cfg in ATTACK_CONFIGS:
            if attack_filter and cfg["label"] not in attack_filter:
                continue
            n_runs = (
                cfg["runs_per_stego"]
                if source_text_type == "stego"
                else cover_runs(cfg["label"], cfg["runs_per_stego"])
            )
            for tampering in cfg["tampering_levels"]:
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
        "original_token_count": source.get("token_count") or count_tokens(original_text),
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

    n_done_now = 0
    n_skipped = 0
    n_errors = 0

    for source, source_text_type, cfg, tampering, run_idx in plan:
        rid = build_record_id(source["id"], cfg["label"], tampering, run_idx)
        if rid in completed:
            n_skipped += 1
            continue

        seed = derive_seed(source["id"], cfg["label"], tampering, run_idx)
        attacked_text, error = attack_one(
            attacks, cfg, source["text"], tampering, seed
        )

        if error:
            n_errors += 1
            log.warning(f"[{system}] {rid} attack failed: {error}")

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
        append_jsonl(out_path, record)
        completed.add(rid)
        n_done_now += 1

        if n_done_now % 25 == 0:
            log.info(
                f"[{system}] progress: {n_done_now} new, {n_skipped} skipped, "
                f"{n_errors} errors (of {len(plan)} planned)"
            )

    log.info(
        f"[{system}] done. wrote {n_done_now} new records "
        f"({n_skipped} skipped, {n_errors} errors)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Apply attacks to Phase 1 texts")
    parser.add_argument(
        "--system",
        choices=[*SYSTEMS, "all"],
        default="all",
        help="Which system(s) to attack",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/experiments"),
        help="Base directory for Phase 1 inputs and Phase 3 outputs",
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
        "--dry-run",
        action="store_true",
        help="Print planned counts without making API calls",
    )
    args = parser.parse_args()

    phase1_dir = args.data_dir / "phase1_texts"
    output_dir = args.data_dir / "phase3_attacks"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        )

    log.info("Phase 3 attacks complete.")


if __name__ == "__main__":
    main()
