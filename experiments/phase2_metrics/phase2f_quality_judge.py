"""Phase 2f: Pairwise text quality — stegotext vs. its normal generation.

Each pair of the detection set (same input, same writer at the same sampling,
same length; ``stegoanalysis_common.load_detection_set``) is shown to LLM
judges from families that wrote none of the texts, in both orders, with the
task input x (LR: with the reference list, which the citation criterion
needs). Judges pick A, B or tie per criterion:

    coherence, relevance, a task criterion (SG: plot consistency;
    LR: citation appropriateness), and overall.

A pair's outcome per criterion combines its two orders: the stegotext wins
when its preferences sum to more than zero (e.g. preferred in one order, tied
in the other), loses below zero, and ties otherwise, so a judge that always
prefers position A scores a tie.

Outputs, under data/experiments/phase2_metrics/detection/quality/:

    {judge}_{system}.jsonl        one record per (pair, order), resumable
    quality_{system}.json         per judge and criterion: win/tie/loss rates,
                                  net win rate with bootstrap CIs over inputs
                                  and a within-pair sign-flip p-value, position
                                  consistency; pooled, per F, per configuration,
                                  per (configuration, F); judge agreement
                                  (Fleiss' kappa) and the majority-vote panel

Usage:
    python -m experiments.phase2_metrics.phase2f_quality_judge --systems story
    python -m experiments.phase2_metrics.phase2f_quality_judge --dry-run
    python -m experiments.phase2_metrics.phase2f_quality_judge --summary-only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.utils.api import chat
from experiments.utils.io import append_jsonl, model_slug, read_jsonl
from experiments.utils.stegoanalysis_common import (
    N_BOOT,
    N_PERM,
    RANDOM_SEED,
    add_common_args,
    detection_metrics_dir,
    load_detection_set,
)
from experiments.utils.system_factory import make_openrouter_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Judges from families that wrote no text (writers: GPT-6, DeepSeek; G: Qwen3.5,
# Gemma 4 — G proposes plot details but writes no text). All on OpenRouter.
JUDGES = ("qwen/qwen3.7-max", "moonshotai/kimi-k3", "z-ai/glm-5.3")

# Room for a judge's (hidden) reasoning before its JSON answer.
MAX_TOKENS = 8000
PARSE_ATTEMPTS = 3

CRITERIA = {
    "story": {
        "coherence": "The story reads as a well-organised whole: events and "
        "ideas connect, transitions are smooth, nothing feels abrupt or out of place.",
        "relevance": "The story develops the given premise.",
        "plot_consistency": "Events, characters and details stay consistent "
        "with each other: no contradictions, and no element that is introduced "
        "and then dropped or left unexplained.",
        "overall": "Overall quality as a short story written for this premise.",
    },
    "litreview": {
        "coherence": "The review reads as a well-organised whole: ideas connect, "
        "transitions are smooth, nothing feels abrupt or out of place.",
        "relevance": "The review covers literature relevant to the paper "
        "described by the title and abstract.",
        "citation_appropriateness": "Each citation supports the claim it is "
        "attached to, and the cited works (see the reference list) fit the "
        "topic where they are cited.",
        "overall": "Overall quality as a related-work section for this paper.",
    },
}

TEXT_KIND = {"story": "story", "litreview": "literature review"}

JUDGE_PROMPT = """You are an expert editor. Two {kind} texts, A and B, were written for the same task input. Compare them on each criterion below.

Task input:
\"\"\"
{task_input}
\"\"\"

Text A:
\"\"\"
{text_a}
\"\"\"

Text B:
\"\"\"
{text_b}
\"\"\"

Criteria:
{criteria}

For each criterion answer "A" if Text A is better, "B" if Text B is better, or "tie" if they are equally good. Judge each criterion on its own, and do not let the order of the texts or their length influence you.

Reply with a JSON object only, with exactly these keys: {keys}. Example: {example}"""


def build_prompt(system: str, task_input: str, text_a: str, text_b: str) -> str:
    crit = CRITERIA[system]
    return JUDGE_PROMPT.format(
        kind=TEXT_KIND[system],
        task_input=task_input,
        text_a=text_a,
        text_b=text_b,
        criteria="\n".join(f"- {k}: {v}" for k, v in crit.items()),
        keys=", ".join(f'"{k}"' for k in crit),
        example=json.dumps({k: v for k, v in zip(crit, ("A", "tie", "B", "A"))}),
    )


def parse_verdicts(system: str, response: str) -> dict[str, str]:
    """The judge's {criterion: "A" | "B" | "tie"}; raises ValueError when the
    reply holds no JSON object with every criterion."""
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", response, re.DOTALL))):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        out = {}
        for k in CRITERIA[system]:
            v = str(obj.get(k, "")).strip().lower()
            if v not in ("a", "b", "tie"):
                break
            out[k] = {"a": "A", "b": "B", "tie": "tie"}[v]
        else:
            return out
    raise ValueError("no JSON verdict with every criterion")


# ---------------------------------------------------------------------------
# Pairs and judging
# ---------------------------------------------------------------------------


def load_pairs(data_dir: Path, system: str, patterns, max_per_cell: int | None):
    """(stego item, normal item) per (cell, input)."""
    by_key: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for it in load_detection_set(data_dir, system, patterns):
        by_key[(it["cell"], it["prompt_idx"])][it["label"]] = it
    pairs = [(v[1], v[0]) for _, v in sorted(by_key.items()) if 0 in v and 1 in v]
    if max_per_cell is not None:
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for s, n in pairs:
            if seen[s["cell"]] < max_per_cell:
                seen[s["cell"]] += 1
                kept.append((s, n))
        pairs = kept
    return pairs


def pair_id(stego: dict) -> str:
    return f"{stego['cell']}/{stego['prompt_idx']}"


ORDERS = ("stego_first", "normal_first")


def judge_one(client, judge: str, system: str, stego: dict, normal: dict, order: str) -> dict:
    a, b = (stego, normal) if order == "stego_first" else (normal, stego)
    prompt = build_prompt(system, stego["input_full"], a["text"], b["text"])
    last_error = None
    for _ in range(PARSE_ATTEMPTS):
        response = chat(
            client,
            judge,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        try:
            verdicts = parse_verdicts(system, response)
            break
        except ValueError as e:
            last_error = e
    else:
        raise ValueError(f"{last_error}; last reply: {response[:300]!r}")
    stego_letter = "A" if order == "stego_first" else "B"
    return {
        "id": f"{pair_id(stego)}/{order}",
        "pair": pair_id(stego),
        "order": order,
        "judge": judge,
        "system": system,
        "cell": stego["cell"],
        "config": stego["config"],
        "F": stego["F"],
        "prompt_idx": stego["prompt_idx"],
        "stego_uid": stego["uid"],
        "normal_uid": normal["uid"],
        "verdicts": verdicts,
        # +1 the stegotext is preferred, -1 its normal generation, 0 tie.
        "stego_pref": {
            k: 0 if v == "tie" else (1 if v == stego_letter else -1)
            for k, v in verdicts.items()
        },
        "raw_response": response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_judge(client, judge: str, system: str, pairs, out_path: Path, workers: int) -> None:
    done = {r["id"] for r in read_jsonl(out_path)}
    tasks = [
        (s, n, o) for s, n in pairs for o in ORDERS if f"{pair_id(s)}/{o}" not in done
    ]
    log.info("[%s] %s: %d judged, %d pending", system, judge, len(done), len(tasks))
    n_ok = n_failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(judge_one, client, judge, system, s, n, o): (s, o)
            for s, n, o in tasks
        }
        for fut in as_completed(futures):
            s, o = futures[fut]
            try:
                append_jsonl(out_path, fut.result())
                n_ok += 1
            except Exception as e:
                n_failed += 1
                log.warning("  %s %s/%s failed: %r", judge, pair_id(s), o, e)
            if (n_ok + n_failed) % 50 == 0:
                log.info("  %s progress: %d / %d", judge, n_ok + n_failed, len(tasks))
    log.info("[%s] %s: %d new, %d failed (rerun to retry)", system, judge, n_ok, n_failed)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

OUTCOMES = ("win", "tie", "loss")


def pair_outcomes(records: list[dict], criterion: str) -> dict[str, dict]:
    """Per pair judged in both orders: its outcome for the stegotext, whether
    the two orders agree, and its group keys."""
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_pair[r["pair"]][r["order"]] = r
    out = {}
    for pid, orders in by_pair.items():
        if not all(o in orders for o in ORDERS):
            continue
        prefs = [orders[o]["stego_pref"][criterion] for o in ORDERS]
        total = sum(prefs)
        r = orders[ORDERS[0]]
        out[pid] = {
            "outcome": "win" if total > 0 else ("loss" if total < 0 else "tie"),
            "net": int(np.sign(total)),
            "consistent": prefs[0] == prefs[1],
            "config": r["config"],
            "F": r["F"],
            "prompt_idx": r["prompt_idx"],
        }
    return out


def outcome_stats(pairs: list[dict], seed: int = RANDOM_SEED) -> dict:
    """Win/tie/loss rates, the net win rate (win - loss) with a 95% bootstrap CI
    over inputs, and a two-sided p-value for net = 0 from flipping each pair's
    outcome sign at random (under "no quality difference" the stegotext and its
    normal generation are exchangeable within a pair)."""
    n = len(pairs)
    if n == 0:
        return {"n_pairs": 0}
    net = np.array([p["net"] for p in pairs], dtype=float)
    out = {"n_pairs": n}
    for o in OUTCOMES:
        out[o] = float(np.mean([p["outcome"] == o for p in pairs]))
    out["net_win"] = float(net.mean())
    if "consistent" in pairs[0]:
        out["position_consistency"] = float(np.mean([p["consistent"] for p in pairs]))

    rng = np.random.default_rng(seed)
    inputs = sorted({p["prompt_idx"] for p in pairs})
    pos = np.array([inputs.index(p["prompt_idx"]) for p in pairs])
    counts = np.stack(
        [np.bincount(rng.integers(0, len(inputs), len(inputs)), minlength=len(inputs))
         for _ in range(N_BOOT)]
    )[:, pos].astype(float)
    boot = (counts * net).sum(1) / counts.sum(1)
    out["net_win_ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    flips = np.where(rng.random((N_PERM, n)) < 0.5, -1.0, 1.0)
    null = np.abs((flips * net).mean(1))
    out["net_win_p"] = float((1 + np.sum(null >= abs(net.mean()) - 1e-12)) / (1 + N_PERM))
    return out


def grouped(pairs: dict[str, dict]) -> dict:
    vals = list(pairs.values())
    by_F, by_config, by_cell = defaultdict(list), defaultdict(list), defaultdict(list)
    for p in vals:
        by_F[p["F"]].append(p)
        by_config[p["config"]].append(p)
        by_cell[(p["config"], p["F"])].append(p)
    return {
        "pooled": outcome_stats(vals),
        "by_F": {str(k): outcome_stats(v) for k, v in sorted(by_F.items())},
        "by_config": {k: outcome_stats(v) for k, v in sorted(by_config.items())},
        "by_config_F": {
            f"{c}/F{F}": outcome_stats(v) for (c, F), v in sorted(by_cell.items())
        },
    }


def fleiss_kappa(ratings: list[list[str]]) -> float | None:
    """Fleiss' kappa over subjects each rated by the same number of raters."""
    if not ratings or len(ratings[0]) < 2:
        return None
    m = len(ratings[0])
    counts = np.array([[row.count(c) for c in OUTCOMES] for row in ratings], float)
    p_j = counts.sum(0) / counts.sum()
    P_i = ((counts**2).sum(1) - m) / (m * (m - 1))
    P_bar, P_e = P_i.mean(), (p_j**2).sum()
    return None if P_e == 1 else float((P_bar - P_e) / (1 - P_e))


def summarize(system: str, records_by_judge: dict[str, list[dict]]) -> dict:
    summary: dict = {"system": system, "judges": {}, "agreement": {}, "panel": {}}
    for criterion in CRITERIA[system]:
        per_judge = {
            j: pair_outcomes(recs, criterion) for j, recs in records_by_judge.items()
        }
        for j, pairs in per_judge.items():
            summary["judges"].setdefault(j, {})[criterion] = grouped(pairs)
        judges = [j for j, p in per_judge.items() if p]
        common = sorted(set.intersection(*(set(per_judge[j]) for j in judges))) if judges else []
        ratings = [[per_judge[j][pid]["outcome"] for j in judges] for pid in common]
        pairwise = [
            float(np.mean([r[a] == r[b] for r in ratings]))
            for a in range(len(judges))
            for b in range(a + 1, len(judges))
        ] if ratings else []
        summary["agreement"][criterion] = {
            "judges": judges,
            "n_pairs": len(common),
            "fleiss_kappa": fleiss_kappa(ratings),
            "mean_pairwise_agreement": float(np.mean(pairwise)) if pairwise else None,
        }
        # Majority vote across judges; no majority counts as a tie.
        panel = {}
        for pid, row in zip(common, ratings):
            top = max(OUTCOMES, key=row.count)
            outcome = top if row.count(top) * 2 > len(row) else "tie"
            panel[pid] = {
                **{k: per_judge[judges[0]][pid][k] for k in ("config", "F", "prompt_idx")},
                "outcome": outcome,
                "net": {"win": 1, "tie": 0, "loss": -1}[outcome],
            }
        summary["panel"][criterion] = grouped(panel)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common_args(parser)
    parser.add_argument(
        "--judges",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(JUDGES),
        help=f"Comma-separated OpenRouter model ids (default: {','.join(JUDGES)}).",
    )
    parser.add_argument(
        "--max-pairs-per-cell",
        type=int,
        default=None,
        help="Judge only the first N inputs of each cell. Default: all.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the call count and one prompt."
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="Only rebuild the summaries."
    )
    args = parser.parse_args()

    out_dir = detection_metrics_dir(args.data_dir) / "quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = None if (args.dry_run or args.summary_only) else make_openrouter_client()

    for system in args.systems:
        pairs = load_pairs(args.data_dir, system, args.dirs, args.max_pairs_per_cell)
        if not pairs:
            log.warning("[%s] no detection pairs found — skipping", system)
            continue
        paths = {j: out_dir / f"{model_slug(j)}_{system}.jsonl" for j in args.judges}
        log.info(
            "[%s] %d pairs x %d orders x %d judges = %d calls",
            system,
            len(pairs),
            len(ORDERS),
            len(args.judges),
            len(pairs) * len(ORDERS) * len(args.judges),
        )
        if args.dry_run:
            s, n = pairs[0]
            print(build_prompt(system, s["input_full"], s["text"][:400], n["text"][:400]))
            continue
        if not args.summary_only:
            for judge in args.judges:
                run_judge(client, judge, system, pairs, paths[judge], args.workers)

        wanted = {pair_id(s) for s, _ in pairs}
        records = {
            j: [r for r in read_jsonl(p) if r["pair"] in wanted] for j, p in paths.items()
        }
        summary = summarize(system, {j: r for j, r in records.items() if r})
        out_path = out_dir / f"quality_{system}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        for j, crit in summary["judges"].items():
            o = crit["overall"]["pooled"]
            if o.get("n_pairs"):
                log.info(
                    "  %s overall: win %.2f tie %.2f loss %.2f (net %+.2f, p=%.3f, "
                    "position consistency %.2f, %d pairs)",
                    j,
                    o["win"],
                    o["tie"],
                    o["loss"],
                    o["net_win"],
                    o["net_win_p"],
                    o["position_consistency"],
                    o["n_pairs"],
                )
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
