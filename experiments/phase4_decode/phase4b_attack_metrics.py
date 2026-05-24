"""Phase 4b: Attack severity metrics (Experiment 1).

For every record in ``data/experiments/phase3_attacks/{system}_attacked.jsonl``
(stego and cover sources alike), compute four metrics comparing the attacked
text to the original text:

    - BERTScore F1 (semantic similarity)
    - Cosine similarity of sentence-transformer embeddings (all-mpnet-base-v2)
    - BLEU  (sacrebleu, sentence-level)
    - TER   (sacrebleu, sentence-level)

Outputs:

    data/experiments/phase4_decode/attack_metrics/{system}_attack_metrics.jsonl
    data/experiments/phase4_decode/attack_metrics_summary.json
    data/experiments/phase4_decode/attack_metrics_summary.tsv

Per-record JSONL is checkpoint-resumable: ids already present are skipped.
BERTScore is computed in batches for speed; cosine / BLEU / TER are per-pair.
A tqdm progress bar is shown per system.

Usage:
    python -m experiments.phase4_decode.phase4b_attack_metrics --system all
    python -m experiments.phase4_decode.phase4b_attack_metrics \
        --system topicqa --batch-size 16 --limit 20  # smoke test
    python -m experiments.phase4_decode.phase4b_attack_metrics \
        --system all --summary-only                  # rebuild summary tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path

import numpy as np
from tqdm import tqdm

from experiments.utils.io import append_jsonl, load_completed_ids, read_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
)
log = logging.getLogger(__name__)

SYSTEMS = ("topicqa", "story", "litreview", "baseline")
SBERT_DEFAULT = "all-mpnet-base-v2"
BERTSCORE_LANG = "en"


# ---------------------------------------------------------------------------
# Lazy model loaders
# ---------------------------------------------------------------------------

_SBERT = None
_SACREBLEU_BLEU = None
_SACREBLEU_TER = None


def _get_sbert(model_name: str):
    global _SBERT
    if _SBERT is None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading SentenceTransformer model: %s", model_name)
        _SBERT = SentenceTransformer(model_name)
    return _SBERT


def _get_sacrebleu_metrics():
    global _SACREBLEU_BLEU, _SACREBLEU_TER
    if _SACREBLEU_BLEU is None:
        import sacrebleu

        _SACREBLEU_BLEU = sacrebleu
        _SACREBLEU_TER = sacrebleu
    return _SACREBLEU_BLEU, _SACREBLEU_TER


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def batched_bertscore(refs: list[str], hyps: list[str], lang: str = BERTSCORE_LANG):
    """Run bert_score.score once over the whole list. Returns list of F1 floats."""
    if not refs:
        return []
    from bert_score import score as bert_score_fn

    _, _, F1 = bert_score_fn(hyps, refs, lang=lang, verbose=False)
    return [float(x.item()) for x in F1]


def batched_cosine(refs: list[str], hyps: list[str], model_name: str) -> list[float]:
    if not refs:
        return []
    model = _get_sbert(model_name)
    emb_ref = model.encode(refs, convert_to_numpy=True, show_progress_bar=False)
    emb_hyp = model.encode(hyps, convert_to_numpy=True, show_progress_bar=False)
    a = emb_ref / (np.linalg.norm(emb_ref, axis=1, keepdims=True) + 1e-12)
    b = emb_hyp / (np.linalg.norm(emb_hyp, axis=1, keepdims=True) + 1e-12)
    return [float(x) for x in (a * b).sum(axis=1)]


def per_pair_bleu_ter(ref: str, hyp: str) -> tuple[float, float]:
    sb_bleu, sb_ter = _get_sacrebleu_metrics()
    bleu_score = sb_bleu.sentence_bleu(hyp, [ref]).score
    ter_score = sb_ter.sentence_ter(hyp, [ref]).score
    return float(bleu_score), float(ter_score)


# ---------------------------------------------------------------------------
# Per-system loop
# ---------------------------------------------------------------------------


def run_system(
    system: str,
    phase3_dir: Path,
    out_dir: Path,
    sbert_model: str,
    batch_size: int,
    limit: int | None,
):
    attack_path = phase3_dir / f"{system}_attacked.jsonl"
    if not attack_path.exists():
        log.warning("[%s] no Phase 3 attacks at %s — skipping", system, attack_path)
        return

    metrics_dir = out_dir / "attack_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"{system}_attack_metrics.jsonl"

    records = read_jsonl(attack_path)
    completed = load_completed_ids(out_path)

    pending: list[dict] = []
    for r in records:
        rid = r.get("id")
        if not rid or rid in completed:
            continue
        if not r.get("attacked_text") or not r.get("original_text"):
            continue
        pending.append(r)

    if limit is not None:
        pending = pending[:limit]

    log.info(
        "[%s] %d total / %d already done / %d pending",
        system,
        len(records),
        len(completed),
        len(pending),
    )
    if not pending:
        return

    n_done = 0
    bar = tqdm(
        total=len(pending),
        desc=f"metrics/{system}",
        unit="rec",
        dynamic_ncols=True,
    )

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        refs = [r["original_text"] for r in batch]
        hyps = [r["attacked_text"] for r in batch]

        try:
            bs_f1 = batched_bertscore(refs, hyps)
        except Exception as e:
            log.warning("[%s] bertscore batch failed: %s — falling back per-record", system, e)
            bs_f1 = []
            for ref, hyp in zip(refs, hyps):
                try:
                    f = batched_bertscore([ref], [hyp])[0]
                except Exception as e2:
                    log.warning("  per-record bertscore failed: %s", e2)
                    f = None
                bs_f1.append(f)

        try:
            cos = batched_cosine(refs, hyps, sbert_model)
        except Exception as e:
            log.warning("[%s] cosine batch failed: %s", system, e)
            cos = [None] * len(batch)

        for r, bf1, cs in zip(batch, bs_f1, cos):
            try:
                bl, tr = per_pair_bleu_ter(r["original_text"], r["attacked_text"])
            except Exception as e:
                log.warning("[%s] %s: bleu/ter failed: %s", system, r["id"], e)
                bl, tr = None, None

            out = {
                "id": r["id"],
                "source_id": r["source_id"],
                "source_text_type": r.get("source_text_type"),
                "system": r["system"],
                "attack_label": r["attack_label"],
                "attack_type": r.get("attack_type"),
                "local": r.get("local"),
                "tampering_level": r["tampering_level"],
                "run_idx": r["run_idx"],
                "original_token_count": r.get("original_token_count"),
                "attacked_token_count": r.get("attacked_token_count"),
                "bertscore_f1": bf1,
                "cosine_similarity": cs,
                "bleu": bl,
                "ter": tr,
            }
            append_jsonl(out_path, out)
            n_done += 1
            bar.update(1)
            bar.set_postfix(done=n_done, refresh=False)

    bar.close()
    log.info("[%s] wrote %d new metric records to %s", system, n_done, out_path)


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


GROUP_FIELDS = ("system", "source_text_type", "attack_label", "tampering_level")
METRIC_FIELDS = ("bertscore_f1", "cosine_similarity", "bleu", "ter")


def _stat_block(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "median": statistics.median(vals),
    }


def build_summary(systems: tuple[str, ...], out_dir: Path) -> dict:
    metrics_dir = out_dir / "attack_metrics"
    grouped: dict[tuple, dict[str, list[float]]] = {}

    for system in systems:
        path = metrics_dir / f"{system}_attack_metrics.jsonl"
        if not path.exists():
            continue
        for r in read_jsonl(path):
            key = tuple(r.get(f) for f in GROUP_FIELDS)
            slot = grouped.setdefault(key, {m: [] for m in METRIC_FIELDS})
            for m in METRIC_FIELDS:
                v = r.get(m)
                if v is not None:
                    slot[m].append(v)

    summary: list[dict] = []
    for key in sorted(
        grouped,
        key=lambda k: (
            str(k[0]),
            str(k[1]),
            str(k[2]),
            float(k[3]) if k[3] is not None else -1.0,
        ),
    ):
        row = dict(zip(GROUP_FIELDS, key))
        for m in METRIC_FIELDS:
            row[m] = _stat_block(grouped[key][m])
        summary.append(row)

    return {"groups": summary}


def write_summary_tsv(summary: dict, tsv_path: Path) -> None:
    rows = summary.get("groups", [])
    headers = list(GROUP_FIELDS) + ["n"]
    for m in METRIC_FIELDS:
        headers += [f"{m}_mean", f"{m}_std"]

    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(headers)
        for row in rows:
            n_val = max((row[m].get("n", 0) for m in METRIC_FIELDS), default=0)
            line = [row.get(g) for g in GROUP_FIELDS] + [n_val]
            for m in METRIC_FIELDS:
                stat = row[m]
                line += [stat.get("mean"), stat.get("std")]
            w.writerow(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4b: BERTScore/cosine/BLEU/TER on Phase 3 attacks."
    )
    parser.add_argument(
        "--system",
        choices=[*SYSTEMS, "all"],
        default="all",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/experiments"),
    )
    parser.add_argument(
        "--subdir",
        default="recovery_test",
        help=(
            "Sub-directory under phase3_attacks/ and phase4_decode/ to read "
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
            "auto-resolves --subdir to '{system}_cap{N}' to match the Phase 1 variant."
        ),
    )
    parser.add_argument(
        "--sbert-model",
        default=SBERT_DEFAULT,
        help=f"Sentence-transformer model for cosine sim (default: {SBERT_DEFAULT}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for BERTScore + sentence-transformer encoding.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap pending records per system (smoke testing).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip computation; just rebuild the summary tables from existing JSONL.",
    )
    args = parser.parse_args()

    if args.capacity is not None:
        if args.system == "all":
            parser.error("--capacity requires --system to be one of topicqa/story/litreview/baseline (not 'all').")
        if args.subdir == "recovery_test":
            args.subdir = f"{args.system}_cap{args.capacity}"
            log.info(f"--capacity set: defaulting --subdir to {args.subdir!r}")

    phase3_dir = args.data_dir / "phase3_attacks"
    out_dir = args.data_dir / "phase4_decode"
    if args.subdir:
        phase3_dir = phase3_dir / args.subdir
        out_dir = out_dir / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Phase 3 inputs: %s", phase3_dir)
    log.info("Phase 4 outputs: %s", out_dir)

    targets = SYSTEMS if args.system == "all" else (args.system,)

    if not args.summary_only:
        for system in targets:
            run_system(
                system=system,
                phase3_dir=phase3_dir,
                out_dir=out_dir,
                sbert_model=args.sbert_model,
                batch_size=args.batch_size,
                limit=args.limit,
            )

    summary = build_summary(targets, out_dir)
    summary_json = out_dir / "attack_metrics_summary.json"
    summary_tsv = out_dir / "attack_metrics_summary.tsv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    write_summary_tsv(summary, summary_tsv)
    log.info("Wrote summary: %s and %s", summary_json, summary_tsv)


if __name__ == "__main__":
    main()
