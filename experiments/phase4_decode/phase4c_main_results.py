"""Phase 4c: Aggregate Phase 4a decode results into the main paper tables.

Reads each system's decoded jsonl from its capacity-tagged subdir
``data/experiments/phase4_decode/{system}_cap{N}/{system}_decoded.jsonl``
and writes the aggregated tables under
``data/experiments/phase4_decode/{output_subdir}/``:

    main_results.json
    main_results_table.tsv
    attack_curves.json
    attack_curves.tsv

The per-system capacity mapping is controlled by ``--capacities`` (see CLI
help). It defaults to the original recovery_test capacities
(topicqa=6, story=18, litreview=20, baseline=3), so the script keeps working
on the data that was migrated out of recovery_test/.

Aggregation procedure (matches experiment.md lines 340-347):

    1. Per (system, source_id, attack_label, tampering_level): mean BER across
       runs — gives one BER value per stego (soaks up attack variance).
    2. Per (system, attack_label, tampering_level): aggregate the per-stego BER
       across stegos — mean, std, plus perfect-recovery rate. Std across stegos
       is the between-stego error bar reported in the paper.

Token efficiency (bits / token) is sourced per-system from
``phase2_metrics/{system}_cap{N}/token_counts_summary.json`` if available,
otherwise computed on the fly from
``phase1_texts/{system}_cap{N}/{system}_stego.jsonl``.

Usage:
    # Default capacities (original recovery_test data):
    python -m experiments.phase4_decode.phase4c_main_results

    # Override a few:
    python -m experiments.phase4_decode.phase4c_main_results \\
        --capacities story=14,litreview=16,topicqa=7 \\
        --output-subdir main_t7_s14_l16_b3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from experiments.utils.io import read_jsonl
from experiments.utils.token_counter import bits_per_token

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
)
log = logging.getLogger(__name__)

SYSTEMS = ("topicqa", "story", "litreview", "baseline")

# Columns shown in the headline main-results table.
HEADLINE_ATTACKS = (
    ("no_attack", 0.0),
    ("global_paraphrase", 1.0),
    ("global_backtranslation", 1.0),
)

# All non-headline attacks that should still appear in attack_curves.* outputs.
CURVE_ATTACKS = (
    "synonym",
    "local_paraphrase",
    "local_backtranslation",
    "global_paraphrase",
    "global_backtranslation",
)


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return statistics.pstdev(values)


def _fmt(v: float | None, places: int = 4) -> str:
    if v is None:
        return ""
    return f"{v:.{places}f}"


# ---------------------------------------------------------------------------
# Token efficiency loader
# ---------------------------------------------------------------------------


def load_token_efficiency(
    data_dir: Path,
    system_subdirs: dict[str, str],
) -> dict[str, dict]:
    """Return {system: {capacity, mean_token_count, bits_per_token}} for stego texts.

    ``system_subdirs`` maps each system to its phase1/phase2 subdir (e.g.
    ``{"story": "story_cap18", "topicqa": "topicqa_cap6"}``). A shared subdir
    or an empty string can be used to read from the top-level dirs.
    """
    out: dict[str, dict] = {}
    for system in SYSTEMS:
        subdir = system_subdirs.get(system, "")
        phase2_dir = data_dir / "phase2_metrics"
        phase1_dir = data_dir / "phase1_texts"
        if subdir:
            phase2_dir = phase2_dir / subdir
            phase1_dir = phase1_dir / subdir
        summary_path = phase2_dir / "token_counts_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    data = json.load(f)
                by_type = data.get(system, {}) if isinstance(data, dict) else {}
                stego = by_type.get("stego", {}) if isinstance(by_type, dict) else {}
                if stego:
                    out[system] = {
                        "capacity_bits": stego.get("capacity_bits"),
                        "mean_token_count": (
                            stego.get("token_count", {}).get("mean")
                            if isinstance(stego.get("token_count"), dict)
                            else stego.get("mean_token_count")
                        ),
                        "bits_per_token": (
                            stego.get("bits_per_token", {}).get("mean")
                            if isinstance(stego.get("bits_per_token"), dict)
                            else stego.get("bits_per_token")
                        ),
                        "source": str(summary_path),
                    }
                    log.info("[%s] token efficiency from %s", system, summary_path)
                    continue
            except Exception as e:
                log.warning("[%s] failed reading %s (%s) — falling back to phase1", system, summary_path, e)

        # Fallback: compute directly from Phase 1 stego records.
        stego_path = phase1_dir / f"{system}_stego.jsonl"
        if not stego_path.exists():
            log.warning("[%s] no Phase 1 stego file at %s — skipping token efficiency", system, stego_path)
            continue
        records = read_jsonl(stego_path)
        if not records:
            continue
        token_counts: list[int] = []
        bits_lengths: list[int] = []
        bpt_values: list[float] = []
        for r in records:
            tc = r.get("token_count")
            mb = r.get("message_bits")
            if not isinstance(tc, int) or not isinstance(mb, list):
                continue
            token_counts.append(tc)
            bits_lengths.append(len(mb))
            bpt_values.append(bits_per_token(len(mb), tc))
        out[system] = {
            "capacity_bits": (
                int(statistics.mode(bits_lengths)) if bits_lengths else None
            ),
            "mean_token_count": _mean(token_counts),
            "bits_per_token": _mean(bpt_values),
            "source": str(stego_path),
        }
        log.info("[%s] token efficiency computed from %s", system, stego_path)
    return out


# ---------------------------------------------------------------------------
# Decode aggregation
# ---------------------------------------------------------------------------


def aggregate_system(decoded_path: Path) -> dict:
    """Two-level aggregation (runs → stegos → cell) for one system's decoded JSONL."""
    if not decoded_path.exists():
        return {"cells": {}, "n_records": 0}

    records = read_jsonl(decoded_path)

    # Level 1: group runs by (attack_label, tampering, source_id) → list of BER.
    per_stego: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    per_stego_perfect: dict[tuple[str, float, str], list[bool]] = defaultdict(list)
    for r in records:
        key = (r["attack_label"], float(r["tampering_level"]), r["source_id"])
        ber = r.get("bit_error_rate")
        if ber is None:
            continue
        per_stego[key].append(float(ber))
        per_stego_perfect[key].append(bool(r.get("perfect_recovery", False)))

    # Level 2: collapse runs → one value per stego, then aggregate over stegos.
    cells: dict[tuple[str, float], dict] = {}
    cell_buckets: dict[tuple[str, float], dict[str, list]] = defaultdict(
        lambda: {"per_stego_ber": [], "per_run_perfect": [], "all_runs_perfect_per_stego": []}
    )
    for (attack_label, tampering, source_id), bers in per_stego.items():
        cell_key = (attack_label, tampering)
        bucket = cell_buckets[cell_key]
        bucket["per_stego_ber"].append(_mean(bers))
        runs_perfect = per_stego_perfect[(attack_label, tampering, source_id)]
        bucket["per_run_perfect"].extend(runs_perfect)
        bucket["all_runs_perfect_per_stego"].append(all(runs_perfect))

    for cell_key, bucket in cell_buckets.items():
        per_stego_ber = bucket["per_stego_ber"]
        per_run_perfect = bucket["per_run_perfect"]
        all_runs_perfect = bucket["all_runs_perfect_per_stego"]
        cells[cell_key] = {
            "n_stegos": len(per_stego_ber),
            "n_runs": len(per_run_perfect),
            "ber_mean": _mean(per_stego_ber),
            "ber_std": _std(per_stego_ber),
            "ber_min": min(per_stego_ber) if per_stego_ber else None,
            "ber_max": max(per_stego_ber) if per_stego_ber else None,
            "perfect_run_rate": (
                sum(per_run_perfect) / len(per_run_perfect) if per_run_perfect else None
            ),
            "perfect_stego_rate": (
                sum(all_runs_perfect) / len(all_runs_perfect)
                if all_runs_perfect
                else None
            ),
        }

    return {"cells": cells, "n_records": len(records)}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def build_main_results(
    systems: tuple[str, ...],
    aggregates: dict[str, dict],
    token_eff: dict[str, dict],
) -> dict:
    rows: list[dict] = []
    for system in systems:
        cells = aggregates.get(system, {}).get("cells", {})
        eff = token_eff.get(system, {})

        row = {
            "system": system,
            "capacity_bits": eff.get("capacity_bits"),
            "mean_token_count": eff.get("mean_token_count"),
            "bits_per_token": eff.get("bits_per_token"),
        }

        for label, tampering in HEADLINE_ATTACKS:
            cell = cells.get((label, tampering), {})
            tag = label
            row[f"{tag}_ber_mean"] = cell.get("ber_mean")
            row[f"{tag}_ber_std"] = cell.get("ber_std")
            row[f"{tag}_perfect_stego_rate"] = cell.get("perfect_stego_rate")
            row[f"{tag}_n_stegos"] = cell.get("n_stegos")
            row[f"{tag}_n_runs"] = cell.get("n_runs")

        rows.append(row)

    return {"systems": systems, "rows": rows}


def write_main_table_tsv(table: dict, path: Path) -> None:
    rows = table["rows"]
    headers = [
        "system",
        "capacity_bits",
        "bits_per_token",
        "mean_token_count",
    ]
    for label, _ in HEADLINE_ATTACKS:
        headers += [
            f"{label}_ber_mean",
            f"{label}_ber_std",
            f"{label}_perfect_stego_rate",
            f"{label}_n_stegos",
            f"{label}_n_runs",
        ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(headers)
        for row in rows:
            line = []
            for h in headers:
                v = row.get(h)
                if h == "system" or h == "capacity_bits":
                    line.append(v if v is not None else "")
                elif isinstance(v, float):
                    line.append(_fmt(v, 4))
                else:
                    line.append("" if v is None else v)
            w.writerow(line)


def build_attack_curves(
    systems: tuple[str, ...],
    aggregates: dict[str, dict],
) -> dict:
    curves: dict[str, dict[str, list[dict]]] = {}
    for system in systems:
        cells = aggregates.get(system, {}).get("cells", {})
        per_attack: dict[str, list[dict]] = {}
        for (label, tampering), cell in cells.items():
            if label not in CURVE_ATTACKS:
                continue
            per_attack.setdefault(label, []).append(
                {
                    "tampering_level": tampering,
                    "ber_mean": cell["ber_mean"],
                    "ber_std": cell["ber_std"],
                    "perfect_run_rate": cell["perfect_run_rate"],
                    "perfect_stego_rate": cell["perfect_stego_rate"],
                    "n_stegos": cell["n_stegos"],
                    "n_runs": cell["n_runs"],
                }
            )
        for label in per_attack:
            per_attack[label].sort(key=lambda d: d["tampering_level"])
        curves[system] = per_attack
    return curves


def write_attack_curves_tsv(curves: dict, path: Path) -> None:
    headers = [
        "system",
        "attack_label",
        "tampering_level",
        "ber_mean",
        "ber_std",
        "perfect_run_rate",
        "perfect_stego_rate",
        "n_stegos",
        "n_runs",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(headers)
        for system, per_attack in curves.items():
            for attack_label, points in per_attack.items():
                for p in points:
                    w.writerow(
                        [
                            system,
                            attack_label,
                            p["tampering_level"],
                            _fmt(p["ber_mean"]),
                            _fmt(p["ber_std"]),
                            _fmt(p["perfect_run_rate"]),
                            _fmt(p["perfect_stego_rate"]),
                            p["n_stegos"],
                            p["n_runs"],
                        ]
                    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _serialize_aggregate(agg: dict) -> dict:
    """Convert the (label, tampering) tuple-keyed cells into JSON-friendly form."""
    out_cells = []
    for (label, tampering), cell in agg.get("cells", {}).items():
        out_cells.append({"attack_label": label, "tampering_level": tampering, **cell})
    out_cells.sort(key=lambda d: (d["attack_label"], d["tampering_level"]))
    return {"n_records": agg.get("n_records", 0), "cells": out_cells}


DEFAULT_CAPACITIES = {
    "topicqa": 6,
    "story": 18,
    "litreview": 20,
    "baseline": 3,
}


def _parse_capacities(raw: str | None) -> dict[str, int]:
    """Parse 'story=14,litreview=16,topicqa=7' into {system: capacity}.

    None/empty returns DEFAULT_CAPACITIES (matches the original recovery_test data).
    Unknown systems are rejected so typos surface immediately.
    """
    if not raw:
        return dict(DEFAULT_CAPACITIES)
    out = dict(DEFAULT_CAPACITIES)
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise argparse.ArgumentTypeError(
                f"--capacities token {token!r} missing '='; expected e.g. 'story=14'."
            )
        sys_name, _, cap_str = token.partition("=")
        sys_name = sys_name.strip()
        if sys_name not in SYSTEMS:
            raise argparse.ArgumentTypeError(
                f"--capacities unknown system {sys_name!r}; choose from {SYSTEMS}."
            )
        try:
            out[sys_name] = int(cap_str)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"--capacities invalid capacity {cap_str!r} for {sys_name!r}: {e}"
            )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4c: aggregate Phase 4a decode results into paper tables."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/experiments"),
    )
    parser.add_argument(
        "--capacities",
        type=_parse_capacities,
        default=None,
        help=(
            "Per-system capacity mapping, e.g. 'story=14,litreview=16,topicqa=7'. "
            "Each system's inputs are read from '{system}_cap{N}/' subdirs of "
            "phase1_texts/, phase2_metrics/ and phase4_decode/. "
            "Missing systems fall back to the original recovery_test capacities: "
            f"{DEFAULT_CAPACITIES}."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help=(
            "Sub-directory under phase4_decode/ to write aggregated outputs "
            "(main_results*, attack_curves*). Defaults to a name derived from --capacities, "
            "e.g. 'main_t6_s18_l20_b3'."
        ),
    )
    parser.add_argument(
        "--systems",
        type=lambda s: tuple(x.strip() for x in s.split(",") if x.strip()),
        default=SYSTEMS,
        help="Comma-separated systems to include.",
    )
    args = parser.parse_args()

    capacities = args.capacities if args.capacities is not None else dict(DEFAULT_CAPACITIES)
    system_subdirs = {sys_name: f"{sys_name}_cap{cap}" for sys_name, cap in capacities.items()}

    if args.output_subdir is None:
        # e.g. main_t6_s18_l20_b3 — short tag using initial letter of each system.
        tag = "_".join(f"{s[0]}{capacities[s]}" for s in SYSTEMS if s in capacities)
        args.output_subdir = f"main_{tag}"
        log.info("--output-subdir not set: defaulting to %r", args.output_subdir)

    out_dir = args.data_dir / "phase4_decode" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Aggregated output dir: %s", out_dir)
    log.info("Per-system input subdirs: %s", system_subdirs)

    token_eff = load_token_efficiency(args.data_dir, system_subdirs)

    aggregates: dict[str, dict] = {}
    for system in tqdm(args.systems, desc="aggregate", unit="sys"):
        sys_subdir = system_subdirs.get(system, "")
        decode_dir = args.data_dir / "phase4_decode"
        if sys_subdir:
            decode_dir = decode_dir / sys_subdir
        path = decode_dir / f"{system}_decoded.jsonl"
        agg = aggregate_system(path)
        aggregates[system] = agg
        log.info(
            "[%s] aggregated %d decoded records into %d (attack, tampering) cells from %s",
            system,
            agg.get("n_records", 0),
            len(agg.get("cells", {})),
            path,
        )

    main_table = build_main_results(args.systems, aggregates, token_eff)
    curves = build_attack_curves(args.systems, aggregates)

    main_json_path = out_dir / "main_results.json"
    main_tsv_path = out_dir / "main_results_table.tsv"
    curves_json_path = out_dir / "attack_curves.json"
    curves_tsv_path = out_dir / "attack_curves.tsv"

    main_payload = {
        "headline_attacks": [
            {"attack_label": label, "tampering_level": tp}
            for label, tp in HEADLINE_ATTACKS
        ],
        "token_efficiency": token_eff,
        "main_table": main_table,
        "aggregates": {s: _serialize_aggregate(a) for s, a in aggregates.items()},
    }
    with open(main_json_path, "w", encoding="utf-8") as f:
        json.dump(main_payload, f, indent=2, default=str)
    write_main_table_tsv(main_table, main_tsv_path)

    with open(curves_json_path, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2, default=str)
    write_attack_curves_tsv(curves, curves_tsv_path)

    log.info("Wrote %s", main_json_path)
    log.info("Wrote %s", main_tsv_path)
    log.info("Wrote %s", curves_json_path)
    log.info("Wrote %s", curves_tsv_path)


if __name__ == "__main__":
    main()
