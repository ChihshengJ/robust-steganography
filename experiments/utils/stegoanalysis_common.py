"""Shared data loading, splits and metrics for the Phase 2 steganalysis detectors.

The detection set is every stegotext of the detection track's cells
(``phase1_texts/detection/{system}_cap{F}_{config}/``, restricted to the
selected inputs) paired with its normal generation: same input, same writer at
the same sampling, same length. The classes differ only in the scheme.

The monitor's view is (task input x, text): x is public, so every detector gets
it next to the text (``task_input``).

The main split is leave-one-configuration-out (``loco_folds``): the
configuration is private and the space of configurations is open, so a
detector must find the pipeline's trace in a configuration it never trained
on. The matched split (``matched_folds``, train and test on one configuration)
is the unrealistic upper bound and runs only when asked.

Detectors write held-out predictions, one row per text per split, so any
breakdown (configuration, F) reads the same files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from experiments.utils.configs import stego_config
from experiments.utils.io import read_jsonl, read_normal_records, read_stego_records

SYSTEMS = ["story", "litreview"]
RANDOM_SEED = 42
N_FOLDS = 5
N_BOOT = 1000
N_PERM = 1000

DEFAULT_EMBEDDER_INSTRUCTION = (
    "Given a task input and an AI-generated text written for it, encode them "
    "to capture stylistic, structural, and content-selection patterns useful "
    "for distinguishing steganographically-encoded text from ordinary "
    "generated text."
)

# {system}_cap{F}_{config}: a detection cell always carries its configuration.
CELL_RE = re.compile(r"^(?P<system>[a-z]+)_cap(?P<F>\d+)_(?P<config>.+)$")

# The writers were given the first 600 characters of the seed abstract
# (GENERATE_REVIEW / GENERATE_REVIEW_NORMAL), so that is the input shown here.
ABSTRACT_CHARS = 600


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def detection_metrics_dir(data_dir: Path) -> Path:
    d = data_dir / "phase2_metrics" / "detection"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stegoanalysis_dir(data_dir: Path) -> Path:
    d = detection_metrics_dir(data_dir) / "stegoanalysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument(
        "--systems",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=SYSTEMS,
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help=(
            "Glob patterns under phase1_texts/ selecting the detection cells "
            "(default: 'detection/{system}_cap*')."
        ),
    )


# ---------------------------------------------------------------------------
# Task input x
# ---------------------------------------------------------------------------


def _litreview_corpus():
    from systems.core.litreview import load_corpus
    from systems.paths import litreview_references

    return load_corpus(*litreview_references())


def task_input(system: str, stego: dict, corpus: list[dict] | None, full: bool) -> str:
    """The public task input of a stegotext, as a detector sees it.

    SG: the premise. LR: the seed paper's title and abstract as the writer saw
    them; ``full`` adds the reference list the writer chose citations from,
    which only a long-context detector (the LLM judge) can take.
    """
    if system == "story":
        return f"Premise: {stego['prompt']}"
    from systems.core.litreview import prepare_references

    paper = corpus[int(stego["system_state"]["corpus_idx"])]
    x = (
        f"Paper title: {paper['title']}\n"
        f"Abstract: {paper.get('abstract', '')[:ABSTRACT_CHARS]}"
    )
    if full:
        refs = "\n".join(
            f"  - {r['author_text']} ({r['year']}). {r['ref_title']}"
            for r in prepare_references(paper["references"])
        )
        x += f"\nReferences:\n{refs}"
    return x


# ---------------------------------------------------------------------------
# Detection set
# ---------------------------------------------------------------------------


def detection_cells(data_dir: Path, system: str, patterns: list[str] | None) -> list[Path]:
    root = data_dir / "phase1_texts"
    patterns = patterns or [f"detection/{system}_cap*"]
    cells: list[Path] = []
    for pattern in patterns:
        for d in sorted(root.glob(pattern)):
            if (
                d.is_dir()
                and CELL_RE.match(d.name)
                and (d / f"{system}_stego.jsonl").exists()
                and d not in cells
            ):
                cells.append(d)
    return cells


def load_detection_set(
    data_dir: Path, system: str, patterns: list[str] | None = None
) -> list[dict]:
    """Every (stegotext, normal generation) pair of the detection cells, as two
    items each: ``uid`` (cell-qualified id, unique across cells), ``label`` (1
    stego, 0 normal), ``text``, ``input`` and ``input_full`` (task_input),
    ``cell``, ``F``, ``config`` (the cell's configuration tag), ``synth_model``,
    ``generator_model`` and ``prompt_idx`` (the input, shared by a pair and by
    the same input at every F). A stegotext without a normal generation is
    left out, with a warning."""
    import logging

    log = logging.getLogger(__name__)
    corpus = _litreview_corpus() if system == "litreview" else None
    items: list[dict] = []
    for d in detection_cells(data_dir, system, patterns):
        m = CELL_RE.match(d.name)
        normals = {r["paired_stego_id"]: r for r in read_normal_records(d, system)}
        stegos = read_stego_records(d, system)
        missing = [s["id"] for s in stegos if s["id"] not in normals]
        if missing:
            log.warning(
                "%s: %d of %d stegotexts have no normal generation (run "
                "phase1_normal); left out: %s",
                d.name,
                len(missing),
                len(stegos),
                ", ".join(missing[:5]),
            )
        for stego in stegos:
            normal = normals.get(stego["id"])
            if normal is None:
                continue
            config = stego_config(stego)
            shared = {
                "cell": d.name,
                "system": system,
                "F": int(m.group("F")),
                "config": m.group("config"),
                "synth_model": config["synth_model"],
                "generator_model": config.get("generator_model"),
                "prompt_idx": int(stego["prompt_idx"]),
                "input": task_input(system, stego, corpus, full=False),
                "input_full": task_input(system, stego, corpus, full=True),
            }
            for rec, label in ((stego, 1), (normal, 0)):
                items.append(
                    {
                        "uid": f"{d.name}/{rec['id']}",
                        "id": rec["id"],
                        "label": label,
                        "text": rec["text"],
                        **shared,
                    }
                )
    return items


def with_input(item: dict, text: str | None = None) -> str:
    """A detector's view of a text: the task input, then the text."""
    return f"{item['input']}\n\nText:\n{item['text'] if text is None else text}"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def matched_folds(items: list[dict], n_splits: int = N_FOLDS) -> Iterator[dict]:
    """Matched-configuration split (the upper bound on a monitor): within each
    configuration, K folds over inputs. Grouping by input keeps a stegotext,
    its normal generation, and the same input at other F on one side of a
    split, so a detector cannot learn an input from one class and meet it
    again in the other. F is pooled in training; test predictions carry F."""
    by_config: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_config[it["config"]].append(i)
    for config, idx in sorted(by_config.items()):
        idx = np.asarray(idx)
        y = np.array([items[i]["label"] for i in idx])
        groups = np.array([items[i]["prompt_idx"] for i in idx])
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
        )
        for k, (tr, te) in enumerate(cv.split(idx, y, groups)):
            yield {
                "split": "matched",
                "train_config": config,
                "fold": k,
                "train": idx[tr],
                "test": idx[te],
            }


# A held-out axis and when a training configuration counts as unseen along it:
# it shares no held-out component with the test configuration.
LOCO_AXES = {
    "writer": lambda c, t: c["synth_model"] != t["synth_model"],
    "generator": lambda c, t: c["generator_model"] != t["generator_model"],
    "both": lambda c, t: (
        c["synth_model"] != t["synth_model"]
        and c["generator_model"] != t["generator_model"]
    ),
}


def loco_folds(items: list[dict]) -> Iterator[dict]:
    """Leave-one-configuration-out, by axis (the main result).

    For each axis (writer, generator G, both) and each test configuration, the
    training set is every configuration that shares none of the held-out
    components with it, e.g. held-out writer: both G models with the other
    writer. Test configurations with the same training set share one model.
    An axis that does not vary (LR has no G) is skipped.

    There is no CV over inputs: the test configuration is the test set, and the
    training configurations cover every input, including the test ones. x is
    public, so a monitor can run its guessed configurations on the very input
    it inspects; and every input appears once in each class of every training
    configuration, so an input carries no label. Early stopping splits the
    training configurations by input (``inner_validation``).
    """
    configs: dict[str, dict] = {}
    by_config: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        configs.setdefault(
            it["config"],
            {"synth_model": it["synth_model"], "generator_model": it["generator_model"]},
        )
        by_config[it["config"]].append(i)
    writers = {c["synth_model"] for c in configs.values()}
    generators = {c["generator_model"] for c in configs.values()}
    varies = {
        "writer": len(writers) > 1,
        "generator": len(generators) > 1,
        "both": len(writers) > 1 and len(generators) > 1,
    }
    for axis, unseen in LOCO_AXES.items():
        if not varies[axis]:
            continue
        groups: dict[tuple, list[str]] = defaultdict(list)
        for test in sorted(configs):
            train = tuple(
                c for c in sorted(configs) if unseen(configs[c], configs[test])
            )
            if train:
                groups[train].append(test)
        for k, (train, tests) in enumerate(sorted(groups.items())):
            yield {
                "split": f"loco_{axis}",
                "train_config": "+".join(train),
                "test_configs": tests,
                "fold": k,
                "train": np.array([i for c in train for i in by_config[c]]),
                "test": np.array([i for c in tests for i in by_config[c]]),
            }


SPLITS = {"loco": loco_folds, "matched": matched_folds}
DEFAULT_SPLIT = "loco"


def inner_validation(
    items: list[dict], train: np.ndarray, frac: float = 0.2, seed: int = RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Split a training fold into (fit, validation) by input, for early
    stopping: the test fold is never used to pick a model."""
    inputs = sorted({items[i]["prompt_idx"] for i in train})
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(inputs) * frac)))
    val_inputs = set(rng.choice(inputs, size=n_val, replace=False).tolist())
    in_val = np.array([items[i]["prompt_idx"] in val_inputs for i in train])
    return train[~in_val], train[in_val]


# ---------------------------------------------------------------------------
# Predictions and metrics
# ---------------------------------------------------------------------------


def prediction_rows(
    items: list[dict], fold: dict, scores: np.ndarray, detector: str
) -> list[dict]:
    """One row per test text of a fold; ``score`` is P(stego)."""
    return [
        {
            "uid": items[i]["uid"],
            "detector": detector,
            "split": fold["split"],
            "train_config": fold["train_config"],
            "fold": fold["fold"],
            "config": items[i]["config"],
            "F": items[i]["F"],
            "prompt_idx": items[i]["prompt_idx"],
            "label": items[i]["label"],
            "score": float(s),
        }
        for i, s in zip(fold["test"], scores)
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def predictions_path(data_dir: Path, detector: str, system: str) -> Path:
    return stegoanalysis_dir(data_dir) / f"{detector}_{system}_predictions.jsonl"


def score_metrics(rows: list[dict], seed: int = RANDOM_SEED) -> dict:
    """Accuracy (at 0.5) and AUC of a set of held-out prediction rows, with

    * 95% CIs from a bootstrap over inputs (a pair, and the same input at every
      F and configuration, share x, so inputs are the independent units);
    * C2ST p-values (Lopez-Paz & Oquab 2017) from permuting the labels within
      each (stegotext, normal) pair: under "same distribution given x" the two
      texts of a pair are exchangeable, so the trained detector's statistic
      should look like a random within-pair swap of the labels. Rows whose
      pair partner is missing (an unparsable judge answer) are left out of the
      test.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0}
    y = np.array([r["label"] for r in rows])
    s = np.array([r["score"] for r in rows], dtype=float)
    out: dict = {"n": n, "accuracy": float(np.mean((s >= 0.5) == y)), "auc": None}
    if len(set(y.tolist())) < 2:
        return out
    pos, neg = s[y == 1], s[y == 0]
    # cmp[i, j]: positive i outranks negative j (ties count half).
    cmp = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    out["auc"] = float(cmp.mean())
    rng = np.random.default_rng(seed)

    # Cluster bootstrap over inputs: row weights = times its input is drawn.
    inputs = sorted({r["prompt_idx"] for r in rows})
    in_pos = np.array([inputs.index(r["prompt_idx"]) for r in rows])
    counts = np.stack(
        [np.bincount(rng.integers(0, len(inputs), len(inputs)), minlength=len(inputs))
         for _ in range(N_BOOT)]
    )[:, in_pos].astype(float)
    correct = ((s >= 0.5) == y).astype(float)
    acc_b = (counts * correct).sum(1) / counts.sum(1)
    w1, w0 = counts[:, y == 1], counts[:, y == 0]
    auc_b = np.einsum("bi,ij,bj->b", w1, cmp, w0) / (w1.sum(1) * w0.sum(1))
    out["accuracy_ci"] = [float(np.percentile(acc_b, 2.5)), float(np.percentile(acc_b, 97.5))]
    out["auc_ci"] = [float(np.nanpercentile(auc_b, 2.5)), float(np.nanpercentile(auc_b, 97.5))]

    # Within-pair label permutation.
    pairs: dict[tuple, dict[int, int]] = defaultdict(dict)
    for k, r in enumerate(rows):
        pairs[(r["config"], r["F"], r["prompt_idx"])][r["label"]] = k
    full = [(p[1], p[0]) for p in pairs.values() if 0 in p and 1 in p]
    if full:
        a = np.array([f[0] for f in full])  # the stegotext of each pair
        b = np.array([f[1] for f in full])  # its normal generation
        pred = (s >= 0.5).astype(float)
        swap = rng.random((N_PERM, len(full))) < 0.5
        # Correct predictions per pair with its labels kept, and swapped.
        keep_acc = pred[a] + (1 - pred[b])
        swap_acc = (1 - pred[a]) + pred[b]
        acc_p = np.where(swap, swap_acc, keep_acc).sum(1) / (2 * len(full))
        obs_acc = keep_acc.sum() / (2 * len(full))
        # AUC over the paired texts, with each pair's labels kept or swapped.
        ranks = _ranks(np.concatenate([s[a], s[b]]))
        m = len(full)
        r_a, r_b = ranks[:m], ranks[m:]
        pos_rank_sum = np.where(swap, r_b, r_a).sum(1)
        auc_p = (pos_rank_sum - m * (m + 1) / 2) / (m * m)
        obs_auc = (r_a.sum() - m * (m + 1) / 2) / (m * m)
        out["n_pairs"] = m
        out["c2st_p_accuracy"] = float((1 + np.sum(acc_p >= obs_acc)) / (1 + N_PERM))
        out["c2st_p_auc"] = float((1 + np.sum(auc_p >= obs_auc)) / (1 + N_PERM))
    return out


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties sharing their mean rank."""
    from scipy.stats import rankdata

    return rankdata(x)


def summarize_predictions(rows: list[dict]) -> dict:
    """Metrics per split: pooled, per F, per test configuration, and per
    (test configuration, F)."""
    out: dict = {}
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    for split, srows in sorted(by_split.items()):
        by_config: dict[str, list[dict]] = defaultdict(list)
        by_cell: dict[tuple, list[dict]] = defaultdict(list)
        by_F: dict[int, list[dict]] = defaultdict(list)
        for r in srows:
            by_config[r["config"]].append(r)
            by_cell[(r["config"], r["F"])].append(r)
            by_F[r["F"]].append(r)
        out[split] = {
            "pooled": score_metrics(srows),
            "by_F": {str(F): score_metrics(v) for F, v in sorted(by_F.items())},
            "by_config": {c: score_metrics(v) for c, v in sorted(by_config.items())},
            "by_config_F": {
                f"{c}/F{F}": score_metrics(v) for (c, F), v in sorted(by_cell.items())
            },
        }
    return out


# ---------------------------------------------------------------------------
# Camera-ready (Type-1 era) data: the n-gram and genre diagnostics only.
# The detectors above never read these.
# ---------------------------------------------------------------------------

SUB_EXP_COVER = {"2a": "cover_c1", "2b": "cover_c2", "2c": "cover_c3"}


def phase1_path(data_dir: Path, system: str, text_type: str) -> Path:
    return data_dir / "phase1_texts" / f"{system}_{text_type}.jsonl"


def load_pair(
    data_dir: Path, system: str, sub_exp: str
) -> tuple[list[dict], list[dict]]:
    """Return (stego_records, cover_records) of the camera-ready top-level
    dataset, sorted by prompt_idx and aligned."""
    cover_type = SUB_EXP_COVER[sub_exp]
    stego = read_jsonl(phase1_path(data_dir, system, "stego"))
    cover = read_jsonl(phase1_path(data_dir, system, cover_type))

    stego_by_idx = {r["prompt_idx"]: r for r in stego if "prompt_idx" in r}
    cover_by_idx = {r["prompt_idx"]: r for r in cover if "prompt_idx" in r}
    common = sorted(set(stego_by_idx) & set(cover_by_idx))
    return [stego_by_idx[i] for i in common], [cover_by_idx[i] for i in common]
