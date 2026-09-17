"""Detection accuracy on the cover texts — the paper's stegoanalysis bar figure.

Plots the two classifiers the figure reports, PPL (a 5-fold logistic regression
on perplexity alone) and Embed (an MLP over text-embedding-005), against the
cover texts, one group per system. Error bars are the standard deviation across
the five folds.

Colors are shared with ``plot_recovery.R`` / ``plot_recovery.py`` so the two
figures read as one system.

**Cover type.** What the paper calls the cover text is ``cover_c2`` for QA and
LR, but ``cover_c3`` for SG: C2's construction was judged too strict for story
generation, and C3 is the straightforward analog there. ``SYSTEM_COVER`` below
is the single place that mapping lives, so the figure never silently mixes
cover constructions.

Reads ``data/experiments/phase2_metrics/stegoanalysis/``:
    summary_{sub}_{system}.json       — PPL (perplexity_only)
    embedding_mlp_{sub}_{system}.json — Embed

Usage:
    python -m experiments.plot_detection
    python -m experiments.plot_detection --out paper/src/figures/detection.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# system -> (sub-experiment holding its cover text, display label)
SYSTEM_COVER = {
    "topicqa": ("2b", "QA"),
    "story": ("2c", "SG"),
    "litreview": ("2b", "LR"),
}

# Shared with the recovery figure's palette.
METHODS = (
    ("PPL", "#15649CFF"),
    ("Embed", "#BB5C33FF"),
)

EMBEDDING_MODEL = "text-embedding-005"
CHANCE = 50.0

BASE_SIZE = 20
PANEL_BORDER = "#333333"
GRID_COLOR = "#EBEBEB"
AXIS_TEXT_COLOR = "#4D4D4D"
MM_TO_PT = 72.27 / 25.4


def _pct(d: dict, key: str) -> tuple[float, float]:
    return d["mean"][key] * 100.0, d["std"][key] * 100.0


def load_results(stego_dir: Path) -> list[dict]:
    """Return one row per (system, method) with mean and fold std, in percent."""
    rows = []
    for system, (sub, label) in SYSTEM_COVER.items():
        summary_path = stego_dir / f"summary_{sub}_{system}.json"
        embed_path = stego_dir / f"embedding_mlp_{sub}_{system}.json"

        ppl = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            block = summary.get("perplexity_only", {})
            if "mean" in block:
                ppl = _pct(block, "accuracy")
            else:
                log.warning(
                    "[%s] no perplexity_only in %s (%s)",
                    system,
                    summary_path.name,
                    block.get("status") or block.get("error"),
                )
        else:
            log.warning("[%s] missing %s", system, summary_path.name)

        embed = None
        if embed_path.exists():
            models = json.loads(embed_path.read_text())["models"]
            if EMBEDDING_MODEL in models:
                embed = _pct(models[EMBEDDING_MODEL], "accuracy")
        else:
            log.warning("[%s] missing %s", system, embed_path.name)

        for method, value in (("PPL", ppl), ("Embed", embed)):
            if value is None:
                log.warning("[%s] no %s result — bar omitted", system, method)
                continue
            rows.append(
                {
                    "system": system,
                    "label": label,
                    "cover": sub,
                    "method": method,
                    "accuracy": value[0],
                    "std": value[1],
                }
            )
    return rows


def build_figure(rows: list[dict], width: float, height: float, chance: bool):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [label for _, (_, label) in SYSTEM_COVER.items()]
    by_key = {(r["label"], r["method"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=(BASE_SIZE / 22) * MM_TO_PT * 0.55)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL_BORDER)
        spine.set_linewidth(0.9)

    n = len(METHODS)
    bar_w = 0.34
    for i, (method, color) in enumerate(METHODS):
        offset = (i - (n - 1) / 2) * bar_w
        xs, ys, errs = [], [], []
        for j, label in enumerate(labels):
            r = by_key.get((label, method))
            if r is None:
                continue
            xs.append(j + offset)
            ys.append(r["accuracy"])
            errs.append(r["std"])
        ax.bar(
            xs,
            ys,
            width=bar_w,
            color=color,
            edgecolor=PANEL_BORDER,
            linewidth=0.6,
            label=method,
            zorder=3,
        )
        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            fmt="none",
            ecolor="#1A1A1A",
            elinewidth=1.4,
            capsize=5,
            capthick=1.4,
            zorder=4,
        )
        for x, y, e in zip(xs, ys, errs):
            ax.text(
                x,
                y + e + 2.0,
                f"{y:.1f}",
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
                zorder=5,
            )

    if chance:
        ax.axhline(
            CHANCE, ls="--", lw=1.2, color="#8A8A8A", zorder=2, dashes=(4, 3)
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Accuracy (%)", fontsize=22, fontweight="bold")
    ax.tick_params(
        colors=PANEL_BORDER, labelcolor=AXIS_TEXT_COLOR, labelsize=18, length=BASE_SIZE / 4
    )
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(METHODS),
        frameon=False,
        fontsize=18,
        title="Method",
        columnspacing=1.4,
        handlelength=1.4,
    )
    legend.get_title().set_fontsize(BASE_SIZE)
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    ap.add_argument("--out", type=Path, default=Path("detection_plot.png"))
    ap.add_argument("--width", type=float, default=7.0)
    ap.add_argument("--height", type=float, default=5.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--no-chance-line", action="store_true", help="drop the 50%% chance rule"
    )
    args = ap.parse_args()

    stego_dir = args.data_dir / "phase2_metrics" / "stegoanalysis"
    rows = load_results(stego_dir)
    if not rows:
        raise SystemExit(f"no results found under {stego_dir}")
    for r in rows:
        log.info(
            "%s (%s, %s): %s = %.1f +/- %.1f",
            r["label"],
            r["system"],
            r["cover"],
            r["method"],
            r["accuracy"],
            r["std"],
        )

    fig = build_figure(rows, args.width, args.height, not args.no_chance_line)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    log.info("wrote figure: %s", args.out)


if __name__ == "__main__":
    main()
