"""Python port of ``plot_recovery.R`` — the main recovery figure.

Same input (``recovery_results.csv`` from
``experiments.phase4_decode.recovery_csv``), same filtering, same facet layout,
and the same output file. Exists so the figure can be regenerated without a
tidyverse install; the R script remains the reference.

Layout it reproduces:

    facet_grid(metric ~ system, scales = "free_x")

rows = {bitwise, perfect} with right-hand strips, columns = systems with top
strips, y shared across every panel, x free per column. ggplot's theme_bw at
base_size 20 is approximated closely enough that the two PNGs are visually
interchangeable.

Usage:
    python -m experiments.plot_recovery
    python -m experiments.plot_recovery --csv recovery_results.csv --out recovery_plot.png
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The R script's scales, verbatim
# --------------------------------------------------------------------------- #

ATTACK_LEVELS = (
    "local_paraphrase",
    "global_paraphrase",
    "synonym",
    "global_backtranslation",
)

# attack_label -> tampering_level to keep; None keeps every level (the global
# attacks only ever have one).
ATTACK_TAMPERING: dict[str, float | None] = {
    "local_paraphrase": 0.5,
    "synonym": 0.5,
    "global_backtranslation": None,
    "global_paraphrase": None,
}

ATTACK_LABELS = {
    "local_paraphrase": "P (local, p=0.5)",
    "synonym": "SS (local, p=0.5)",
    "global_backtranslation": "RTT (global)",
    "global_paraphrase": "P (global)",
}

COLORS = {
    "synonym": "#BB5C33FF",
    "global_paraphrase": "#15649CFF",
    "global_backtranslation": "#E3D6BBFF",
    "local_paraphrase": "#7296B8FF",
}

SYSTEM_LEVELS = ("topicqa", "story", "litreview", "discop")
SYSTEM_LABELS = {"topicqa": "QA", "story": "SG", "litreview": "LR", "discop": "Discop"}

METRICS = (("bit-wise_accuracy", "bitwise"), ("perfect_stego_rate", "perfect"))

MAX_CAPACITY = 20  # R: filter(capacity < 20)
Y_BREAKS = (0.2, 0.4, 0.6, 0.8, 1.0)

# theme_bw(base_size = 20), as ggplot renders it
BASE_SIZE = 20
PANEL_BORDER = "#333333"  # grey20
GRID_COLOR = "#EBEBEB"  # grey92
STRIP_FILL = "#D9D9D9"  # grey85
AXIS_TEXT_COLOR = "#4D4D4D"  # grey30
MM_TO_PT = 72.27 / 25.4  # ggplot's .pt: linewidth/size units are mm


def _expand(lo: float, hi: float, mult: float = 0.05) -> tuple[float, float]:
    """ggplot's default continuous expansion."""
    if hi == lo:
        return lo - 0.5, hi + 0.5
    pad = (hi - lo) * mult
    return lo - pad, hi + pad


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_long(csv_path: Path) -> list[dict]:
    """Read the CSV, apply the R filter, and pivot the two metrics to long form."""
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            attack = r["attack_label"]
            if attack not in ATTACK_TAMPERING:
                continue
            keep_t = ATTACK_TAMPERING[attack]
            if keep_t is not None and float(r["tampering_level"]) != keep_t:
                continue
            capacity = int(r["capacity"])
            if capacity >= MAX_CAPACITY:
                continue
            system = r["system"]
            if system not in SYSTEM_LEVELS:
                log.warning("dropping unknown system %r (not a facet)", system)
                continue
            for column, metric in METRICS:
                rows.append(
                    {
                        "system": system,
                        "attack_label": attack,
                        "capacity": capacity,
                        "metric": metric,
                        "accuracy": float(r[column]),
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #


def build_figure(long: list[dict], width: float, height: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    systems = [s for s in SYSTEM_LEVELS if any(r["system"] == s for r in long)]
    metrics = [label for _, label in METRICS]
    if not systems:
        raise SystemExit("no plottable rows — check the CSV's system column")

    # series[(system, metric, attack)] -> {capacity: accuracy}
    series: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for r in long:
        series[(r["system"], r["metric"], r["attack_label"])][r["capacity"]] = r["accuracy"]

    # facet_grid: y is shared across every panel, x is free per column.
    ys = [r["accuracy"] for r in long]
    ylim = _expand(min(ys), max(ys))
    xlims, xticks = {}, {}
    for s in systems:
        caps = sorted({r["capacity"] for r in long if r["system"] == s})
        xticks[s] = caps
        xlims[s] = _expand(min(caps), max(caps))

    fig, axes = plt.subplots(
        len(metrics),
        len(systems),
        figsize=(width, height),
        squeeze=False,
        sharey=True,
    )
    fig.patch.set_facecolor("white")

    for i, metric in enumerate(metrics):
        for j, system in enumerate(systems):
            ax = axes[i][j]
            ax.set_facecolor("white")
            ax.set_axisbelow(True)
            ax.grid(
                True, color=GRID_COLOR, linewidth=(BASE_SIZE / 22) * MM_TO_PT * 0.55
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(PANEL_BORDER)
                spine.set_linewidth(0.9)

            for attack in ATTACK_LEVELS:
                points = series.get((system, metric, attack))
                if not points:
                    continue
                xs = sorted(points)
                ax.plot(
                    xs,
                    [points[x] for x in xs],
                    color=COLORS[attack],
                    linewidth=1.1 * MM_TO_PT,
                    marker="o",
                    markersize=3 * MM_TO_PT,
                    markeredgewidth=0,
                    solid_capstyle="butt",
                    zorder=3,
                )

            ax.set_xlim(*xlims[system])
            ax.set_ylim(*ylim)
            ax.set_xticks(xticks[system])
            if i != len(metrics) - 1:
                ax.tick_params(labelbottom=False)
            ax.set_yticks([b for b in Y_BREAKS if ylim[0] <= b <= ylim[1]])
            ax.tick_params(
                colors=PANEL_BORDER,
                labelcolor=AXIS_TEXT_COLOR,
                labelsize=18,
                length=BASE_SIZE / 4,
            )
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontweight("bold")

            # facet strips: systems on top, metrics on the right.
            if i == 0:
                ax.set_title(
                    SYSTEM_LABELS[system],
                    fontsize=BASE_SIZE,
                    fontweight="bold",
                    backgroundcolor=STRIP_FILL,
                    bbox=dict(
                        facecolor=STRIP_FILL, edgecolor=PANEL_BORDER, linewidth=0.9, pad=6
                    ),
                    pad=10,
                )
            if j == len(systems) - 1:
                sec = ax.secondary_yaxis("right")
                sec.set_yticks([])
                for spine in sec.spines.values():
                    spine.set_visible(False)
                sec.set_ylabel(
                    metric,
                    fontsize=BASE_SIZE,
                    fontweight="bold",
                    rotation=-90,
                    va="bottom",
                    labelpad=22,
                    bbox=dict(
                        facecolor=STRIP_FILL, edgecolor=PANEL_BORDER, linewidth=0.9, pad=6
                    ),
                )

    fig.supxlabel("Message Length", fontsize=22, fontweight="bold", y=0.09)
    fig.supylabel("Accuracy", fontsize=22, fontweight="bold")

    handles = [
        Line2D(
            [],
            [],
            color=COLORS[a],
            linewidth=1.1 * MM_TO_PT,
            marker="o",
            markersize=3 * MM_TO_PT,
            markeredgewidth=0,
            label=ATTACK_LABELS[a],
        )
        for a in ATTACK_LEVELS
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.0),
        ncol=len(handles),
        frameon=False,
        fontsize=18,
        columnspacing=0.8,
        handlelength=1.4,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.tight_layout(rect=(0.015, 0.155, 1, 1))
    # tight_layout also reserves room for supxlabel, which is already placed by
    # hand above the legend — reclaim it so the panels are not floated upward.
    fig.subplots_adjust(bottom=0.205)

    # ggplot puts a bottom legend's title to the left of the keys, vertically
    # centred. matplotlib only stacks it above, so place it by hand once the
    # legend's extent is known.
    fig.canvas.draw()
    box = legend.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(
        box.x0 - 0.012,
        (box.y0 + box.y1) / 2,
        "attack",
        fontsize=BASE_SIZE,
        fontweight="bold",
        ha="right",
        va="center",
    )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("recovery_results.csv"))
    ap.add_argument("--out", type=Path, default=Path("recovery_plot.png"))
    ap.add_argument("--width", type=float, default=12.0)
    ap.add_argument("--height", type=float, default=6.0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"missing {args.csv} — build it with "
            "`python -m experiments.phase4_decode.recovery_csv`"
        )

    long = load_long(args.csv)
    if not long:
        raise SystemExit(f"no rows survived the filter in {args.csv}")
    log.info("%d plotted values from %s", len(long), args.csv)

    fig = build_figure(long, args.width, args.height)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    log.info("wrote figure: %s", args.out)


if __name__ == "__main__":
    main()
