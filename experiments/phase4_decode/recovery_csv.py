"""Compile the Phase 4a decode outputs into the flat CSV that ``plot_recovery.R`` reads.

One row per (run, attack_label, tampering_level), with the two accuracy metrics
the figure plots. Aggregation matches ``phase4c_main_results`` and
``baseline_capacity_sweep``: per stego, average bitwise accuracy over its attack
runs (soaking up attack variance), then average over stegos. Perfect recovery is
reported both per run and per stego (all runs perfect), and the figure uses the
per-stego rate.

Runs are discovered from ``data/experiments/phase4_decode/{run}/{system}_decoded.jsonl``
where ``run`` is ``{system}_cap{m}`` plus an optional suffix (``_len575_sp`` for the
length-matched SyncPool baselines). ``--runs`` overrides the default selection.

Usage:
    python -m experiments.phase4_decode.recovery_csv
    python -m experiments.phase4_decode.recovery_csv --out recovery_results.csv
    python -m experiments.phase4_decode.recovery_csv --runs topicqa_cap6,discop_cap16_len575_sp
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RUN_RE = re.compile(r"^(?P<system>[a-z0-9]+)_cap(?P<m>\d+)(?P<suffix>.*)$")

# The runs the main recovery figure is built from: our three systems at their
# swept message lengths, plus the length-matched SyncPool Discop baseline.
DEFAULT_RUNS = (
    "topicqa_cap6",
    "topicqa_cap8",
    "topicqa_cap10",
    "story_cap14",
    "story_cap16",
    "story_cap18",
    "litreview_cap14",
    "litreview_cap16",
    "litreview_cap18",
    "litreview_cap20",
    "discop_cap14_len575_sp",
    "discop_cap16_len575_sp",
    "discop_cap18_len575_sp",
)

FIELDS = (
    "system",
    "run",
    "capacity",
    "attack_label",
    "tampering_level",
    "n_stegos",
    "n_records",
    "n_errors",
    "bit-wise_accuracy",
    "perfect_run_rate",
    "perfect_stego_rate",
)


def load_records(path: Path) -> list[dict]:
    """Read a decoded jsonl, dropping duplicate ids (phase-1 reruns append)."""
    out: list[dict] = []
    seen: set[str] = set()
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
    return out


def aggregate_run(records: list[dict]) -> list[dict]:
    """Return one aggregated cell per (attack_label, tampering_level)."""
    cells: dict[tuple[str, float], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        if r.get("bit_error_rate") is None:
            continue
        key = (r["attack_label"], float(r.get("tampering_level", 0.0)))
        cells[key][r["source_id"]].append(r)

    rows = []
    for (attack, tampering), by_src in sorted(cells.items()):
        # Step 1: per-stego mean over runs. Step 2: mean over stegos.
        stego_acc = [
            sum(1.0 - x["bit_error_rate"] for x in runs) / len(runs)
            for runs in by_src.values()
        ]
        all_runs = [bool(x.get("perfect_recovery")) for runs in by_src.values() for x in runs]
        stego_perfect = [
            all(bool(x.get("perfect_recovery")) for x in runs) for runs in by_src.values()
        ]
        rows.append(
            {
                "attack_label": attack,
                "tampering_level": tampering,
                "n_stegos": len(by_src),
                "n_records": len(all_runs),
                "n_errors": sum(
                    1 for runs in by_src.values() for x in runs if x.get("error")
                ),
                "bit-wise_accuracy": sum(stego_acc) / len(stego_acc),
                "perfect_run_rate": sum(all_runs) / len(all_runs),
                "perfect_stego_rate": sum(stego_perfect) / len(stego_perfect),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    ap.add_argument("--out", type=Path, default=Path("recovery_results.csv"))
    ap.add_argument(
        "--runs",
        default=",".join(DEFAULT_RUNS),
        help="comma list of phase4_decode subdir names, or 'all' to discover them",
    )
    args = ap.parse_args()

    decode_dir = args.data_dir / "phase4_decode"
    if args.runs == "all":
        runs = sorted(p.name for p in decode_dir.iterdir() if p.is_dir())
    else:
        runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    rows: list[dict] = []
    for run in runs:
        mt = RUN_RE.match(run)
        if not mt:
            log.warning("skipping %s: not a {system}_cap{m} run", run)
            continue
        system, capacity = mt.group("system"), int(mt.group("m"))
        path = decode_dir / run / f"{system}_decoded.jsonl"
        if not path.exists():
            log.warning("missing decode file: %s", path)
            continue
        records = load_records(path)
        cells = aggregate_run(records)
        for cell in cells:
            rows.append({"system": system, "run": run, "capacity": capacity, **cell})
        log.info("[%s] %d records -> %d cells", run, len(records), len(cells))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log.info("wrote %d rows: %s", len(rows), args.out)


if __name__ == "__main__":
    main()
