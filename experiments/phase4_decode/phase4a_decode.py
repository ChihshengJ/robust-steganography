"""Phase 4a: Decode attacked stego texts and compute BER.

Reads ``data/experiments/phase3_attacks/{system}_attacked.jsonl``, restores
each stego's system state, calls ``system.recover_message(attacked_text)``,
and writes one JSONL record per (source_id, attack_label, tampering, run_idx)
to ``data/experiments/phase4_decode/{system}_decoded.jsonl``.

Optionally also decodes the *original* (unattacked) stego texts as a
``no_attack`` baseline so Phase 4c can report "BER (no attack)" without a
separate run. Enabled by default; turn off with ``--no-baseline``.

The original ``message_bits`` are looked up from
``data/experiments/phase1_texts/{system}_stego.jsonl`` by ``source_id``.

Output schema:

    {
      "id": "topicqa_s_000_global_paraphrase_1.0_run0",
      "source_id": "topicqa_s_000",
      "system": "topicqa",
      "attack_label": "global_paraphrase",
      "attack_type": "paraphrase",
      "local": false,
      "tampering_level": 1.0,
      "run_idx": 0,
      "original_bits": [...],
      "recovered_bits": [...],
      "bitwise_accuracy": 0.833,
      "bit_error_rate": 0.167,
      "perfect_recovery": false,
      "num_bit_errors": 1,
      "error": null
    }

Resumable: ids already present in the output JSONL are skipped on resume.
Usage:
    python -m experiments.phase4_decode.phase4a_decode --system all
    python -m experiments.phase4_decode.phase4a_decode --system topicqa
    python -m experiments.phase4_decode.phase4a_decode \
        --system story --attack global_paraphrase --limit 10  # smoke test
    python -m experiments.phase4_decode.phase4a_decode --system all --dry-run
    python -m experiments.phase4_decode.phase4a_decode --system topicqa --no-baseline
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from experiments.utils.io import (
    append_jsonl,
    load_completed_ids,
    load_records_map,
    read_jsonl,
)
from experiments.utils.metrics import bit_error_rate
from experiments.utils.system_factory import (
    make_baseline,
    make_clients,
    make_litreview,
    make_story,
    make_topicqa,
    restore_system_state,
)
from systems import StegSystem

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
)
log = logging.getLogger(__name__)

SYSTEMS = ("topicqa", "story", "litreview", "baseline")


# ---------------------------------------------------------------------------
# System construction
# ---------------------------------------------------------------------------


def build_system(
    system: str,
    client,
    local_client,
    n_subtopics: int = 12,
    group_size: int = 2,
    n_slots: int = 20,
):
    if system == "topicqa":
        return make_topicqa(
            client, local_client, n_subtopics=n_subtopics, group_size=group_size
        )
    if system == "story":
        return make_story(client, local_client, n_slots=n_slots)
    if system == "litreview":
        return make_litreview(client)
    if system == "baseline":
        return make_baseline(client)
    raise ValueError(f"Unknown system: {system}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def baseline_id(source_id: str) -> str:
    return f"{source_id}_no_attack_0_run0"


def make_baseline_record(stego_rec: dict) -> dict:
    """Build a synthetic 'attack' record that points at the original stego text."""
    return {
        "id": baseline_id(stego_rec["id"]),
        "source_id": stego_rec["id"],
        "source_text_type": "stego",
        "system": stego_rec["system"],
        "attack_label": "no_attack",
        "attack_type": "no_attack",
        "local": False,
        "tampering_level": 0.0,
        "run_idx": 0,
        "original_text": stego_rec["text"],
        "attacked_text": stego_rec["text"],
        "system_state": stego_rec.get("system_state"),
    }


def decode_one(
    system_obj: StegSystem,
    attacked_text: str,
    state: dict,
    expected_len: int,
) -> tuple[list[int] | None, str | None]:
    """Restore state and run recover_message. Returns (bits, error_str_or_None)."""
    if not attacked_text:
        return None, "empty attacked_text"
    try:
        restore_system_state(system_obj, state or {})
        recovered = system_obj.recover_message(attacked_text)
    except Exception as e:
        return None, repr(e)

    if recovered is None:
        return None, "recover_message returned None"

    bits = list(recovered)
    if len(bits) < expected_len:
        bits = bits + [0] * (expected_len - len(bits))
    elif len(bits) > expected_len:
        bits = bits[:expected_len]
    return bits, None


# ---------------------------------------------------------------------------
# Per-system loop
# ---------------------------------------------------------------------------


def run_system(
    system: str,
    client,
    local_client,
    phase1_dir: Path,
    phase3_dir: Path,
    output_dir: Path,
    attack_filter: set[str] | None,
    limit: int | None,
    include_baseline: bool,
    dry_run: bool,
    n_subtopics: int = 12,
    group_size: int = 2,
    n_slots: int = 20,
):
    attack_path = phase3_dir / f"{system}_attacked.jsonl"
    stego_path = phase1_dir / f"{system}_stego.jsonl"
    out_path = output_dir / f"{system}_decoded.jsonl"

    if not attack_path.exists():
        log.warning("[%s] no Phase 3 attacks at %s — skipping", system, attack_path)
        return
    if not stego_path.exists():
        log.warning("[%s] no Phase 1 stegos at %s — skipping", system, stego_path)
        return

    # Map source_id → original phase1 stego record (for original_bits and baseline texts)
    stego_by_id = load_records_map(stego_path)
    log.info("[%s] loaded %d Phase 1 stego records", system, len(stego_by_id))

    # Phase 3 attack records, filtered to stego sources only
    attack_records = [
        r for r in read_jsonl(attack_path) if r.get("source_text_type") == "stego"
    ]
    if attack_filter:
        attack_records = [
            r for r in attack_records if r["attack_label"] in attack_filter
        ]
    log.info(
        "[%s] %d attacked stego records to decode (after filter)",
        system,
        len(attack_records),
    )

    # Build the work plan: optional baselines + attacks
    plan: list[dict] = []
    if include_baseline and not attack_filter:
        # Only emit a baseline per *unique* source_id we'll touch.
        seen_sources = sorted({r["source_id"] for r in attack_records})
        for sid in seen_sources:
            stego_rec = stego_by_id.get(sid)
            if stego_rec is not None:
                plan.append(make_baseline_record(stego_rec))
    plan.extend(attack_records)

    if limit is not None:
        plan = plan[:limit]

    log.info("[%s] %d decode tasks planned (output: %s)", system, len(plan), out_path)

    if dry_run:
        for rec in plan[:3]:
            log.info(
                "  e.g. %s (attack=%s tp=%s run=%s)",
                rec["id"],
                rec["attack_label"],
                rec["tampering_level"],
                rec["run_idx"],
            )
        if len(plan) > 3:
            log.info("  ... and %d more", len(plan) - 3)
        return

    completed = load_completed_ids(out_path)
    log.info("[%s] %d records already done; resuming", system, len(completed))

    pending = [r for r in plan if r["id"] not in completed]
    if not pending:
        log.info("[%s] nothing to do", system)
        return

    # Build the system once per system run
    system_obj = build_system(
        system,
        client,
        local_client,
        n_subtopics=n_subtopics,
        group_size=group_size,
        n_slots=n_slots,
    )

    n_decoded = 0
    n_errors = 0
    n_perfect = 0

    bar = tqdm(
        pending,
        desc=f"decode/{system}",
        unit="rec",
        dynamic_ncols=True,
    )
    for rec in bar:
        source_id = rec["source_id"]
        stego_rec = stego_by_id.get(source_id)
        if stego_rec is None:
            log.warning("[%s] no Phase 1 record for source_id=%s", system, source_id)
            continue

        original_bits = stego_rec.get("message_bits")
        if not original_bits:
            log.warning("[%s] %s has no message_bits — skipping", system, source_id)
            continue

        state = rec.get("system_state") or stego_rec.get("system_state") or {}
        attacked_text = rec.get("attacked_text")

        recovered, err = decode_one(
            system_obj=system_obj,
            attacked_text=attacked_text,
            state=state,
            expected_len=len(original_bits),
        )

        if err is not None or recovered is None:
            n_errors += 1
            ber_block = {
                "ber": 1.0,
                "bitwise_accuracy": 0.0,
                "num_errors": len(original_bits),
                "perfect": False,
            }
            recovered_for_log = None
        else:
            ber_block = bit_error_rate(original_bits, recovered)
            recovered_for_log = recovered
            if ber_block["perfect"]:
                n_perfect += 1

        out_rec = {
            "id": rec["id"],
            "source_id": source_id,
            "system": system,
            "attack_label": rec["attack_label"],
            "attack_type": rec.get("attack_type", rec["attack_label"]),
            "local": rec.get("local", False),
            "tampering_level": rec["tampering_level"],
            "run_idx": rec["run_idx"],
            "original_bits": list(original_bits),
            "recovered_bits": recovered_for_log,
            "bitwise_accuracy": ber_block["bitwise_accuracy"],
            "bit_error_rate": ber_block["ber"],
            "perfect_recovery": ber_block["perfect"],
            "num_bit_errors": ber_block["num_errors"],
            "error": err,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_jsonl(out_path, out_rec)
        completed.add(rec["id"])
        n_decoded += 1

        bar.set_postfix(
            done=n_decoded,
            err=n_errors,
            perfect=n_perfect,
            refresh=False,
        )

    bar.close()
    log.info(
        "[%s] done. decoded=%d errors=%d perfect=%d",
        system,
        n_decoded,
        n_errors,
        n_perfect,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4a: Decode attacked stego texts and compute BER."
    )
    parser.add_argument(
        "--system",
        choices=[*SYSTEMS, "all"],
        default="all",
        help="Which system(s) to decode for.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/experiments"),
        help="Base directory for inputs and outputs.",
    )
    parser.add_argument(
        "--subdir",
        default="recovery_test",
        help=(
            "Sub-directory under phase1_texts/, phase3_attacks/ and "
            "phase4_decode/ to read inputs from and write outputs to "
            "(default: recovery_test). Pass --subdir '' to use the top-level dirs. "
            "If --capacity is set and --subdir is left at the default, "
            "subdir auto-becomes '{system}_cap{N}'."
        ),
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help=(
            "Override message-bit count for the chosen system. Required to match the "
            "Phase 1 variant being decoded. Auto-sets --subdir to '{system}_cap{N}' "
            "unless --subdir is given explicitly. For topicqa, also sets "
            "n_subtopics = capacity * group_size (unless --n-subtopics is given)."
        ),
    )
    parser.add_argument(
        "--n-subtopics",
        type=int,
        default=None,
        help="TopicQA only: must match Phase 1 (default: 12, or 2*capacity).",
    )
    parser.add_argument(
        "--n-slots",
        type=int,
        default=None,
        help="Story only: must match Phase 1 (default: 20).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=2,
        help="TopicQA only: must match Phase 1 (default: 2).",
    )
    parser.add_argument(
        "--attack",
        action="append",
        default=None,
        help=(
            "Filter to one or more attack_labels (repeatable). "
            "Disables baseline injection. Default: all attacks + baseline."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the planned decode tasks per system (smoke testing).",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not decode the original (unattacked) stego texts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned counts without making API calls.",
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

    # Resolve per-system capacity overrides for system reconstruction.
    topicqa_n_subtopics = args.n_subtopics
    if (
        topicqa_n_subtopics is None
        and args.capacity is not None
        and args.system == "topicqa"
    ):
        topicqa_n_subtopics = args.capacity * args.group_size
    if topicqa_n_subtopics is None:
        topicqa_n_subtopics = 12

    story_n_slots = args.n_slots if args.n_slots is not None else 20

    phase1_dir = args.data_dir / "phase1_texts"
    phase3_dir = args.data_dir / "phase3_attacks"
    output_dir = args.data_dir / "phase4_decode"
    if args.subdir:
        phase1_dir = phase1_dir / args.subdir
        phase3_dir = phase3_dir / args.subdir
        output_dir = output_dir / args.subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Phase 1 inputs: %s", phase1_dir)
    log.info("Phase 3 inputs: %s", phase3_dir)
    log.info("Phase 4 outputs: %s", output_dir)

    attack_filter = set(args.attack) if args.attack else None
    include_baseline = not args.no_baseline

    if args.dry_run:
        client = local_client = None
    else:
        client, local_client = make_clients()

    targets = SYSTEMS if args.system == "all" else (args.system,)
    for system in targets:
        run_system(
            system=system,
            client=client,
            local_client=local_client,
            phase1_dir=phase1_dir,
            phase3_dir=phase3_dir,
            output_dir=output_dir,
            attack_filter=attack_filter,
            limit=args.limit,
            include_baseline=include_baseline,
            dry_run=args.dry_run,
            n_subtopics=topicqa_n_subtopics,
            group_size=args.group_size,
            n_slots=story_n_slots,
        )

    log.info("Phase 4a decode complete.")


if __name__ == "__main__":
    main()
