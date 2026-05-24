"""Phase 2a: Token / word counts and token efficiency.

Reads all Phase 1 JSONL outputs at ``data/experiments/phase1_texts/`` and
emits:

    data/experiments/phase2_metrics/token_counts.jsonl
    data/experiments/phase2_metrics/token_counts_summary.json
    data/experiments/phase2_metrics/token_counts.csv

For every text it writes ``{token_count, word_count, char_count}`` and, for
stego texts, ``bits_per_token = len(message_bits) / token_count``.

Token counts are recomputed via ``tiktoken`` (``o200k_base``, GPT-4.1's
tokenizer). If the recomputed value disagrees with what Phase 1 stored,
the recomputed value wins and a warning is logged.

Usage:
    python -m experiments.phase2_metrics.phase2a_token_counts
    python -m experiments.phase2_metrics.phase2a_token_counts --systems topicqa,story
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

from experiments.utils.io import append_jsonl, read_jsonl
from experiments.utils.token_counter import bits_per_token, count_tokens, count_words

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEMS = ["topicqa", "story", "litreview", "baseline"]
TEXT_TYPES = ["stego", "cover_c1", "cover_c2"]


def _phase1_path(data_dir: Path, system: str, text_type: str) -> Path:
    return data_dir / "phase1_texts" / f"{system}_{text_type}.jsonl"


def _build_record(rec: dict) -> dict:
    text = rec["text"]
    tok = count_tokens(text)
    wc = count_words(text)
    cc = len(text)

    stored_tok = rec.get("token_count")
    if stored_tok is not None and stored_tok != tok:
        log.warning(
            "token_count mismatch for %s: stored=%s recomputed=%s — using recomputed",
            rec["id"], stored_tok, tok,
        )

    out = {
        "id": rec["id"],
        "system": rec["system"],
        "text_type": rec["text_type"],
        "prompt_idx": rec.get("prompt_idx"),
        "token_count": tok,
        "word_count": wc,
        "char_count": cc,
    }
    msg_bits = rec.get("message_bits")
    if rec["text_type"] == "stego" and msg_bits is not None:
        out["message_bits_len"] = len(msg_bits)
        out["bits_per_token"] = bits_per_token(len(msg_bits), tok)
    else:
        out["message_bits_len"] = None
        out["bits_per_token"] = None
    return out


def _stat_block(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def _summarize(records: list[dict]) -> dict:
    by_group: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        key = (r["system"], r["text_type"])
        by_group.setdefault(key, []).append(r)

    summary: dict[str, dict] = {}
    for (system, text_type), group in by_group.items():
        block = {
            "n": len(group),
            "token_count": _stat_block([r["token_count"] for r in group]),
            "word_count": _stat_block([r["word_count"] for r in group]),
            "char_count": _stat_block([r["char_count"] for r in group]),
        }
        if text_type == "stego":
            bpt = [r["bits_per_token"] for r in group if r["bits_per_token"] is not None]
            block["bits_per_token"] = _stat_block(bpt)
        summary.setdefault(system, {})[text_type] = block
    return summary


def _write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "system", "text_type", "prompt_idx",
        "token_count", "word_count", "char_count",
        "message_bits_len", "bits_per_token",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2a: token / word counts")
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument(
        "--systems",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=SYSTEMS,
        help="Comma-separated subset of: topicqa,story,litreview",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Append to token_counts.jsonl instead of rewriting from scratch.",
    )
    args = parser.parse_args()

    out_dir = args.data_dir / "phase2_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "token_counts.jsonl"
    summary_path = out_dir / "token_counts_summary.json"
    csv_path = out_dir / "token_counts.csv"

    if not args.no_overwrite and jsonl_path.exists():
        jsonl_path.unlink()

    all_records: list[dict] = []
    for system in args.systems:
        for text_type in TEXT_TYPES:
            src = _phase1_path(args.data_dir, system, text_type)
            phase1_records = read_jsonl(src)
            if not phase1_records:
                log.warning("No records found at %s", src)
                continue
            log.info("[%s/%s] %d records", system, text_type, len(phase1_records))
            for rec in phase1_records:
                out = _build_record(rec)
                append_jsonl(jsonl_path, out)
                all_records.append(out)

    summary = _summarize(all_records)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _write_csv(all_records, csv_path)

    log.info("Wrote %d records to %s", len(all_records), jsonl_path)
    log.info("Wrote summary to %s", summary_path)
    log.info("Wrote CSV to %s", csv_path)


if __name__ == "__main__":
    main()
