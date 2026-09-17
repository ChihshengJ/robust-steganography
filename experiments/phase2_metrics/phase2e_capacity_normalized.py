"""Phase 2e: normalized payload-capacity reporting (revision item E1).

Aggregates *existing* Phase-1 texts and Phase-4 decodes into one normalized
capacity table. No generation, no API calls.

Definitions (see paper Sec. 5, "Payload accounting"):

  m               nominal payload, |h|, in bits per document
  m_native        bits actually pushed through the embedder before repetition
                  coding (token-level baselines only; == m for our schemes)
  r               repetition factor, m_native / m (1 for our schemes)
  |s|_w, |s|_t    stegotext length in whitespace words / o200k_base tokens
  R_w = m/|s|_w   NET rate, bits per word      <- headline normalization
  R_t = m/|s|_t   NET rate, bits per token
  R_w^nat         GROSS rate, m_native/|s|_w   (baselines' advertised rate)
  G_w(a) = R_w * PR(a)   goodput under attack a, using PERFECT recovery rate

Rates are computed per document and then averaged (E[m/|s|]), which is what the
bootstrap CI is over. The ratio-of-means m/E[|s|] is also emitted as `*_agg`
because it is what a reader reconstructs from a mean-length column; the two are
close but not equal and the paper should say which it reports.

Outputs (under data/experiments/phase2_metrics/):
    capacity_per_doc.csv          one row per stegotext
    capacity_by_condition.csv     one row per (system, m) condition
    capacity_goodput.csv          one row per (system, m, attack)
    capacity_summary.json
    capacity_table.tex            main-paper table body

Usage:
    python -m experiments.phase2_metrics.phase2e_capacity_normalized
    python -m experiments.phase2_metrics.phase2e_capacity_normalized --recompute-tokens
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OURS = ("topicqa", "story", "litreview")
BASELINES = ("discop",)

TASK_LABEL = {
    "topicqa": r"\qa",
    "story": r"\sg",
    "litreview": r"\lr",
    "discop": "Discop",
}

CAP_DIR_RE = re.compile(r"^(?P<system>[a-z]+)_cap(?P<m>\d+)(?P<suffix>.*)$")


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def bca_ci(x: np.ndarray, stat=np.mean, n_boot: int = 1000, alpha: float = 0.05,
           seed: int = 0) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI (E4 asks for BCa everywhere)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    theta_hat = stat(x)
    boots = np.array([stat(x[rng.integers(0, n, n)]) for _ in range(n_boot)])

    # bias correction
    prop = np.mean(boots < theta_hat)
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    from scipy.stats import norm  # noqa: PLC0415
    z0 = norm.ppf(prop)

    # acceleration via jackknife
    jack = np.array([stat(np.delete(x, i)) for i in range(n)])
    jmean = jack.mean()
    num = np.sum((jmean - jack) ** 3)
    den = 6.0 * (np.sum((jmean - jack) ** 2) ** 1.5)
    a = 0.0 if den == 0 else num / den

    def adj(q):
        zq = norm.ppf(q)
        return norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))

    lo, hi = np.quantile(boots, [adj(alpha / 2), adj(1 - alpha / 2)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out, seen = [], set()
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["id"] in seen:          # phase-1 reruns appended duplicates
            continue
        seen.add(r["id"])
        out.append(r)
    return out


def discover_conditions(phase1_dir: Path) -> list[dict]:
    conds = []
    for d in sorted(p for p in phase1_dir.iterdir() if p.is_dir()):
        mt = CAP_DIR_RE.match(d.name)
        if not mt:
            continue
        system = mt.group("system")
        stego = d / f"{system}_stego.jsonl"
        if not stego.exists():
            continue
        conds.append({
            "run": d.name,
            "system": system,
            "m": int(mt.group("m")),
            "variant": mt.group("suffix").lstrip("_") or "main",
            "path": stego,
        })
    return conds


def decode_source_ids(decode_dir: Path, run: str, system: str) -> set[str] | None:
    p = decode_dir / run / f"{system}_decoded.jsonl"
    if not p.exists():
        return None
    return {json.loads(l)["source_id"] for l in p.open() if l.strip()}


def perfect_rates(decode_dir: Path, run: str, system: str) -> dict[str, dict]:
    """attack_label -> {perfect_stego_rate, perfect_run_rate, bitwise, n_stegos}."""
    p = decode_dir / run / f"{system}_decoded.jsonl"
    if not p.exists():
        return {}
    per: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for line in p.open():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("bit_error_rate") is None:
            continue
        per[r["attack_label"]][r["source_id"]].append(r)

    out = {}
    for attack, by_src in per.items():
        stego_perfect = [all(bool(x.get("perfect_recovery")) for x in runs)
                         for runs in by_src.values()]
        all_runs = [bool(x.get("perfect_recovery")) for runs in by_src.values() for x in runs]
        stego_acc = [float(np.mean([1.0 - x["bit_error_rate"] for x in runs]))
                     for runs in by_src.values()]
        out[attack] = {
            "n_stegos": len(by_src),
            "perfect_stego_rate": float(np.mean(stego_perfect)),
            "perfect_run_rate": float(np.mean(all_runs)),
            "bitwise_accuracy": float(np.mean(stego_acc)),
            "_stego_perfect": stego_perfect,
        }
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="E1: normalized payload capacity")
    ap.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tex-out", type=Path, default=Path("paper/src/tables/capacity.tex"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--include-variants", default="main",
                    help="comma list of variants to include: main,naive,... or 'all'")
    ap.add_argument("--recompute-tokens", action="store_true",
                    help="re-tokenize with tiktoken o200k_base instead of trusting phase-1")
    ap.add_argument("--restrict-to-decoded", action="store_true", default=True,
                    help="keep only stegotexts that were actually decoded in phase 4")
    args = ap.parse_args()

    phase1 = args.data_dir / "phase1_texts"
    phase4 = args.data_dir / "phase4_decode"
    out_dir = args.out_dir or (args.data_dir / "phase2_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    count_tokens = None
    if args.recompute_tokens:
        from experiments.utils.token_counter import count_tokens  # noqa: PLC0415

    wanted = None if args.include_variants == "all" else set(
        v.strip() for v in args.include_variants.split(",") if v.strip())

    per_doc: list[dict] = []
    by_cond: list[dict] = []
    goodput: list[dict] = []

    for cond in discover_conditions(phase1):
        if wanted is not None and cond["variant"] not in wanted:
            continue
        recs = _read_jsonl(cond["path"])
        if not recs:
            log.warning("empty condition: %s", cond["run"])
            continue

        keep = decode_source_ids(phase4, cond["run"], cond["system"])
        if args.restrict_to_decoded and keep:
            before = len(recs)
            recs = [r for r in recs if r["id"] in keep]
            if len(recs) != before:
                log.warning("%s: dropped %d stegotext(s) with no phase-4 decode "
                            "(kept %d)", cond["run"], before - len(recs), len(recs))

        m = cond["m"]
        for r in recs:
            md = r.get("metadata") or {}
            ss = r.get("system_state") or {}
            tok = count_tokens(r["text"]) if count_tokens else r["token_count"]
            wc = r["word_count"]
            native = md.get("n_payload_bits") or m
            per_doc.append({
                "run": cond["run"], "system": cond["system"], "variant": cond["variant"],
                "id": r["id"], "m_bits": m, "m_native_bits": native,
                "repetitions": ss.get("repetitions") or 1,
                "words": wc, "tokens": tok,
                "bits_per_word": m / wc, "bits_per_token": m / tok,
                "native_bits_per_word": native / wc, "native_bits_per_token": native / tok,
            })

        rows = [d for d in per_doc if d["run"] == cond["run"]]
        w = np.array([d["words"] for d in rows], float)
        t = np.array([d["tokens"] for d in rows], float)
        rw = np.array([d["bits_per_word"] for d in rows], float)
        rt = np.array([d["bits_per_token"] for d in rows], float)
        rw_lo, rw_hi = bca_ci(rw, n_boot=args.n_boot)
        rt_lo, rt_hi = bca_ci(rt, n_boot=args.n_boot)
        rec = {
            "run": cond["run"], "system": cond["system"], "variant": cond["variant"],
            "m_bits": m, "n": len(rows),
            "m_native_bits": rows[0]["m_native_bits"],
            "repetitions": rows[0]["repetitions"],
            "words_mean": w.mean(), "words_sd": w.std(ddof=1), "words_median": np.median(w),
            "tokens_mean": t.mean(), "tokens_sd": t.std(ddof=1), "tokens_median": np.median(t),
            "bits_per_word": rw.mean(), "bits_per_word_lo": rw_lo, "bits_per_word_hi": rw_hi,
            "bits_per_token": rt.mean(), "bits_per_token_lo": rt_lo, "bits_per_token_hi": rt_hi,
            "bits_per_word_agg": m / w.mean(), "bits_per_token_agg": m / t.mean(),
            "native_bits_per_word": float(np.mean([d["native_bits_per_word"] for d in rows])),
            "native_bits_per_token": float(np.mean([d["native_bits_per_token"] for d in rows])),
        }
        by_cond.append(rec)

        # goodput per attack: delivered bits/word = R_w * P(perfect)
        for attack, st in perfect_rates(phase4, cond["run"], cond["system"]).items():
            pr = st["perfect_stego_rate"]
            g = rw * pr
            g_lo, g_hi = bca_ci(np.array(
                [a * b for a, b in zip(rw, st["_stego_perfect"])], float
            ), n_boot=args.n_boot) if len(rw) == len(st["_stego_perfect"]) else (float("nan"),) * 2
            goodput.append({
                "run": cond["run"], "system": cond["system"], "m_bits": m,
                "attack": attack, "n_stegos": st["n_stegos"],
                "bitwise_accuracy": st["bitwise_accuracy"],
                "perfect_stego_rate": pr, "perfect_run_rate": st["perfect_run_rate"],
                "goodput_bits_per_doc": m * pr,
                "goodput_bits_per_word": float(g.mean()),
                "goodput_bits_per_word_lo": g_lo, "goodput_bits_per_word_hi": g_hi,
                "goodput_bits_per_token": float((rt * pr).mean()),
            })

    _write_csv(out_dir / "capacity_per_doc.csv", per_doc)
    _write_csv(out_dir / "capacity_by_condition.csv", by_cond)
    _write_csv(out_dir / "capacity_goodput.csv", goodput)
    with (out_dir / "capacity_summary.json").open("w") as f:
        json.dump({"by_condition": by_cond, "goodput": goodput}, f, indent=2)

    _write_tex(args.tex_out, by_cond, goodput)
    log.info("wrote %d per-doc rows, %d conditions, %d goodput rows",
             len(per_doc), len(by_cond), len(goodput))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in rows[0] if not k.startswith("_")]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s", path)


def _write_tex(path: Path, by_cond: list[dict], goodput: list[dict]) -> None:
    gp = {(g["run"], g["attack"]): g for g in goodput}
    order = {s: i for i, s in enumerate(OURS + BASELINES)}
    rows = sorted(by_cond, key=lambda r: (order.get(r["system"], 99), r["m_bits"]))

    lines = [
        r"% auto-generated by experiments.phase2_metrics.phase2e_capacity_normalized",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"System & $m$ & $|\mathbf{s}|_w$ & $|\mathbf{s}|_t$ & $R_w$ & $R_t$ "
        r"& $R_w^{\mathrm{gross}}$ & $G_w$ \\",
        r" & (bits) & (words) & (tok.) & (b/word) & (b/tok.) & (b/word) & (b/word) \\",
        r"\midrule",
    ]
    prev = None
    for r in rows:
        if prev is not None and r["system"] != prev:
            lines.append(r"\midrule")
        prev = r["system"]
        g = gp.get((r["run"], "global_paraphrase"))
        gtxt = f"{g['goodput_bits_per_word']:.4f}" if g else "--"
        gross = (f"{r['native_bits_per_word']:.2f}"
                 if r["m_native_bits"] != r["m_bits"] else "--")
        lines.append(
            f"{TASK_LABEL.get(r['system'], r['system'])} & {r['m_bits']} & "
            f"{r['words_mean']:.0f}\\,$\\pm$\\,{r['words_sd']:.0f} & "
            f"{r['tokens_mean']:.0f}\\,$\\pm$\\,{r['tokens_sd']:.0f} & "
            f"{r['bits_per_word']:.4f} & {r['bits_per_token']:.4f} & {gross} & {gtxt} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
