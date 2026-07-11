"""Aggregate Meteor/Discop recovery across message lengths into a table + figure.

The paper's recovery figure has *message length* on the x-axis and *accuracy* on
the y-axis, with two lines: bitwise accuracy and perfect-recovery rate. This
script produces exactly that shape for the token-level baselines by reading the
Phase 4a decode outputs at several capacities:

    data/experiments/phase4_decode/{system}_cap{N}/{system}_decoded.jsonl

Aggregation matches phase4c: for each stego, average bitwise accuracy across its
attack runs (soaking up attack variance), then average across stegos. Perfect
recovery is reported the same way (per-run rate and all-runs-perfect-per-stego
rate). One row per (message length) for the chosen attack.

Usage:
    python -m experiments.phase4_decode.baseline_capacity_sweep \\
        --system meteor --capacities 14,16,18 --attack global_paraphrase

    # No-attack ceiling (should be ~100% / 100%):
    python -m experiments.phase4_decode.baseline_capacity_sweep \\
        --system discop --capacities 14,16,18 --attack no_attack
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _decoded_path(data_dir: Path, system: str, cap: int) -> Path:
    return (
        data_dir
        / "phase4_decode"
        / f"{system}_cap{cap}"
        / f"{system}_decoded.jsonl"
    )


def aggregate_one(path: Path, attack_label: str) -> dict | None:
    """Aggregate a single decoded jsonl for one attack_label.

    Returns dict with mean bitwise accuracy, perfect-run rate, and
    perfect-stego rate (all in [0,1]), or None if no matching records.
    """
    if not path.exists():
        log.warning("missing decode file: %s", path)
        return None

    # Group runs per stego: source_id -> lists of per-run metrics.
    per_stego_acc: dict[str, list[float]] = defaultdict(list)
    per_stego_perfect: dict[str, list[bool]] = defaultdict(list)
    n = 0
    for line in path.open():
        r = json.loads(line)
        if r.get("attack_label") != attack_label:
            continue
        n += 1
        src = r["source_id"]
        ber = r.get("bit_error_rate")
        if ber is None:
            continue
        per_stego_acc[src].append(1.0 - ber)  # bitwise accuracy
        per_stego_perfect[src].append(bool(r.get("perfect_recovery", False)))

    if not per_stego_acc:
        log.warning("no '%s' records in %s", attack_label, path)
        return None

    # Step 1: per-stego mean over runs. Step 2: mean over stegos.
    stego_mean_acc = [sum(v) / len(v) for v in per_stego_acc.values()]
    bitwise_accuracy = sum(stego_mean_acc) / len(stego_mean_acc)

    all_runs = [p for runs in per_stego_perfect.values() for p in runs]
    perfect_run_rate = sum(all_runs) / len(all_runs)
    stego_all_perfect = [all(runs) for runs in per_stego_perfect.values()]
    perfect_stego_rate = sum(stego_all_perfect) / len(stego_all_perfect)

    return {
        "n_records": n,
        "n_stegos": len(per_stego_acc),
        "bitwise_accuracy": bitwise_accuracy,
        "perfect_run_rate": perfect_run_rate,
        "perfect_stego_rate": perfect_stego_rate,
    }


def write_table(rows: list[dict], out_tsv: Path) -> None:
    headers = [
        "message_length",
        "n_stegos",
        "bitwise_accuracy",
        "perfect_run_rate",
        "perfect_stego_rate",
    ]
    with out_tsv.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(headers)
        for row in rows:
            w.writerow([row[h] for h in headers])
    log.info("wrote table: %s", out_tsv)


def write_figure(rows: list[dict], out_png: Path, system: str, attack: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib optional
        log.warning("matplotlib unavailable (%s); skipping figure", e)
        return

    xs = [r["message_length"] for r in rows]
    bitwise = [100 * r["bitwise_accuracy"] for r in rows]
    perfect = [100 * r["perfect_stego_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, bitwise, marker="o", label="Bitwise accuracy")
    ax.plot(xs, perfect, marker="s", label="Perfect recovery")
    ax.axhline(50, ls="--", lw=0.8, color="grey", label="Chance (bitwise)")
    ax.set_xlabel("Message length (bits)")
    ax.set_ylabel("Recovery accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks(xs)
    ax.set_title(f"{system} — {attack}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    log.info("wrote figure: %s", out_png)


def main():
    parser = argparse.ArgumentParser(
        description="Message-length recovery sweep for Meteor/Discop baselines."
    )
    parser.add_argument("--system", choices=["meteor", "discop"], required=True)
    parser.add_argument(
        "--capacities",
        default="14,16,18",
        help="Comma-separated message lengths (default: 14,16,18).",
    )
    parser.add_argument(
        "--attack",
        default="global_paraphrase",
        help="attack_label to aggregate (default: global_paraphrase; "
        "use 'no_attack' for the ceiling).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: phase4_decode/{system}_capsweep/).",
    )
    args = parser.parse_args()

    caps = [int(c) for c in args.capacities.split(",") if c.strip()]
    out_dir = args.out_dir or (args.data_dir / "phase4_decode" / f"{args.system}_capsweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cap in caps:
        agg = aggregate_one(_decoded_path(args.data_dir, args.system, cap), args.attack)
        if agg is None:
            continue
        rows.append({"message_length": cap, **agg})

    if not rows:
        log.error("no data aggregated; run phases 1/3/4a at the requested capacities first.")
        return

    tag = args.attack
    write_table(rows, out_dir / f"{args.system}_{tag}_sweep.tsv")
    (out_dir / f"{args.system}_{tag}_sweep.json").write_text(json.dumps(rows, indent=2))
    write_figure(rows, out_dir / f"{args.system}_{tag}_sweep.png", args.system, tag)

    # Echo the table to stdout for convenience.
    print(f"\n{args.system}  attack={tag}")
    print("msg_len\tbitwise_acc\tperfect_run\tperfect_stego\tn_stegos")
    for r in rows:
        print(
            f"{r['message_length']}\t{r['bitwise_accuracy']:.3f}\t\t"
            f"{r['perfect_run_rate']:.3f}\t\t{r['perfect_stego_rate']:.3f}\t\t{r['n_stegos']}"
        )


if __name__ == "__main__":
    main()
