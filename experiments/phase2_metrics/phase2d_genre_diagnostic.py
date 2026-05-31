"""Phase 2d — quantitative genre / style diagnostic for the story system.

Characterizes the three story text classes on genre / style axes. The C2 cover
(Qwen outline + GPT-4.1 synthesis) holds the plot-origin model constant with the
stego pipeline, so stego and C2 should share a "Qwen-shaped" content fingerprint
(thriller-leaning, distinctive character names) — this diagnostic gives concrete
numbers for how close stego sits to each of its covers (C1, C2) on those axes.

Reports, per text class (stego / c1 / c2):

  1. Dark-genre lexicon density — fraction of tokens drawn from a curated
     violence/death/suspense/threat word list. Higher = more thriller-like.
  2. VADER sentiment — mean compound polarity and the `neg` component.
  3. Character-name fingerprints — names extracted via title patterns
     ("Dr./Mr./Ms./Detective ..."), with top-name lists, distinct-name counts,
     and a Jaccard overlap of each class's top-K names vs C2.
  4. TF-IDF + logistic regression on stego vs C2 and stego vs C1 — what 1- and
     2-grams most discriminate the regimes (paper-ready discriminating-phrase list).

For 1 and 2 every contrast (stego vs c2, stego vs c1, c1 vs c2) gets a Welch
t-test, Mann-Whitney U, and Cohen's d so the paper can cite p-values + effect sizes.

Usage:
    python -m experiments.phase2_metrics.phase2d_genre_diagnostic
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from experiments.utils.io import read_jsonl
from experiments.utils.stegoanalysis_common import RANDOM_SEED, phase1_path


# Curated dark-genre lexicon. Drawn from thriller / horror / crime registers.
# Kept conservative: only words whose presence in a story is a real genre
# signal (i.e. "dark" is in but neutral words like "night" are not).
DARK_LEXICON = {
    # death / killing
    "kill", "killed", "killing", "kills", "killer",
    "murder", "murdered", "murderer", "murderous",
    "dead", "death", "deaths", "deadly", "die", "died", "dying",
    "corpse", "corpses", "body", "bodies",
    "blood", "bloody", "bloodied", "bleeding", "bleed",
    # violence
    "stab", "stabbed", "shot", "shoot", "shooting", "gun", "guns",
    "knife", "knives", "weapon", "weapons", "bullet", "bullets",
    "wound", "wounded", "wounds", "scar", "scars", "scream", "screamed",
    "screaming", "screams", "shriek", "shrieked", "shrieking",
    "attack", "attacked", "attacker", "assault", "violence", "violent",
    # threat / fear
    "fear", "feared", "fearful", "afraid", "terror", "terrified",
    "terrifying", "horror", "horrors", "horrified", "horrifying",
    "panic", "panicked", "dread", "dreaded", "dreading",
    "threat", "threats", "threaten", "threatened", "threatening",
    "danger", "dangerous", "menace", "menacing", "ominous",
    "sinister", "malevolent", "evil", "wicked",
    # darkness / shadow
    "dark", "darker", "darkness", "shadow", "shadows", "shadowy",
    "black", "blackness", "gloom", "gloomy", "bleak", "grim",
    # suspense / disaster
    "scream", "haunt", "haunted", "haunting", "stalk", "stalked", "stalker",
    "hunt", "hunted", "hunting", "trap", "trapped", "chase", "chased",
    "escape", "escaped", "escaping", "flee", "fled", "fleeing",
    "disaster", "catastrophe", "catastrophic", "doom", "doomed",
    "tragedy", "tragic", "demise", "perish", "perished",
    # body trauma
    "bruise", "bruised", "broken", "shattered", "shattering",
    "trembling", "tremble", "trembled", "shudder", "shuddered",
    "gasp", "gasped", "gasping", "choke", "choked", "choking",
    "pulse", "pulses", "heartbeat", "heartbeats",
    # crime / pursuit
    "victim", "victims", "suspect", "suspects", "crime", "criminal",
    "investigate", "investigator", "detective", "interrogate",
    "interrogation", "witness",
    # noir setting
    "alley", "alleys", "alleyway", "rotted", "rotting", "decay", "decayed",
    "rust", "rusted", "rusting", "abandoned", "derelict",
}

TITLE_PATTERN = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Miss|Sir|Lady|Lord|Father|Sister|Brother|"
    r"Captain|Cpt|Lieutenant|Lt|Sergeant|Sgt|Officer|Detective|Det|"
    r"Inspector|Agent|Colonel|Major|General|Admiral|Commander|Cmdr|"
    r"President|Senator|Governor|Mayor|Judge|Professor|Prof|Director|"
    r"Chief|King|Queen|Prince|Princess|Duke|Duchess)\.?\s+"
    r"((?:[A-Z][a-z'’\-]+)(?:\s+[A-Z][a-z'’\-]+){0,2})"
)

# Forename pattern for dialogue-attribution fallback: "<Name> said/whispered/..."
DIALOGUE_VERBS = (
    r"said|asked|whispered|shouted|murmured|replied|answered|nodded|"
    r"snapped|hissed|growled|muttered|called|yelled|screamed|gasped|"
    r"breathed|stammered|laughed|sighed|smiled|cried"
)
DIALOGUE_PATTERN = re.compile(
    rf"\b([A-Z][a-z'’\-]+)\s+(?:{DIALOGUE_VERBS})\b"
)
WORD_PATTERN = re.compile(r"[A-Za-z']+")

# Common English words capitalized at sentence start that we never want to
# count as names. (Stop-set for the dialogue fallback only.)
NAME_STOPLIST = {
    "He", "She", "It", "They", "We", "I", "You", "The", "A", "An",
    "His", "Her", "Their", "Our", "My", "Your",
    "But", "And", "Yet", "So", "Or", "For", "Nor",
    "Then", "Now", "Today", "Tomorrow", "Yesterday",
    "Yes", "No", "Maybe", "Perhaps", "Indeed",
    "Outside", "Inside", "Above", "Below", "Behind",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "Subject",
}


def tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in WORD_PATTERN.findall(text)]


def dark_density(text: str) -> float:
    """Fraction of tokens drawn from the curated dark-genre lexicon."""
    toks = tokenize_words(text)
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in DARK_LEXICON)
    return hits / len(toks)


def extract_names(text: str) -> list[str]:
    """Title-prefixed names ("Dr. Eliza Marin") + dialogue-attribution names
    ("Mara whispered"). Keeps only the first name token (so 'Dr. Eliza Marin'
    contributes 'Eliza', and dialogue 'Mara whispered' contributes 'Mara')."""
    names = []

    for m in TITLE_PATTERN.finditer(text):
        full = m.group(1).strip()
        first = full.split()[0]
        names.append(first)

    for m in DIALOGUE_PATTERN.finditer(text):
        cand = m.group(1)
        if cand in NAME_STOPLIST:
            continue
        names.append(cand)

    return names


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled SD (ddof=1)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def contrast(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> dict:
    """Welch t-test + Mann-Whitney + Cohen's d for two samples."""
    t_res = stats.ttest_ind(a, b, equal_var=False)
    u_res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "contrast": f"{name_a} vs {name_b}",
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a, ddof=1)),
        "std_b": float(np.std(b, ddof=1)),
        "welch_t": float(t_res.statistic),
        "welch_p": float(t_res.pvalue),
        "mannwhitney_u": float(u_res.statistic),
        "mannwhitney_p": float(u_res.pvalue),
        "cohens_d": cohen_d(a, b),
    }


def load_texts(data_dir: Path, text_type: str) -> list[str]:
    path = phase1_path(data_dir, "story", text_type)
    records = read_jsonl(path)
    records.sort(key=lambda r: r.get("prompt_idx", 0))
    return [r["text"] for r in records if r.get("text")]


def per_text_metrics(texts: list[str], sia: SentimentIntensityAnalyzer) -> dict:
    dark = np.array([dark_density(t) for t in texts])
    sent = [sia.polarity_scores(t) for t in texts]
    compound = np.array([s["compound"] for s in sent])
    neg = np.array([s["neg"] for s in sent])
    pos = np.array([s["pos"] for s in sent])
    return {
        "dark": dark,
        "compound": compound,
        "neg": neg,
        "pos": pos,
    }


def name_distribution(texts: list[str]) -> tuple[Counter, int, int]:
    """Return (name counter, total mentions, distinct-name count)."""
    counter: Counter = Counter()
    total = 0
    for t in texts:
        for n in extract_names(t):
            counter[n] += 1
            total += 1
    return counter, total, len(counter)


def jaccard_topk(a: Counter, b: Counter, k: int) -> float:
    top_a = {n for n, _ in a.most_common(k)}
    top_b = {n for n, _ in b.most_common(k)}
    if not top_a and not top_b:
        return float("nan")
    return len(top_a & top_b) / len(top_a | top_b)


def tfidf_discriminators(
    texts_pos: list[str],
    texts_neg: list[str],
    label_pos: str,
    label_neg: str,
    top_k: int = 20,
) -> dict:
    """Fit TF-IDF (1-2 gram, stopwords kept) + logistic regression on
    (pos vs neg). Returns CV accuracy and the top discriminating n-grams
    each way."""
    docs = texts_pos + texts_neg
    y = np.array([1] * len(texts_pos) + [0] * len(texts_neg))

    # CV accuracy with per-fold vectorizer (no leakage).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    accs = []
    arr = np.array(docs, dtype=object)
    for tr, te in skf.split(arr, y):
        vec = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=3, sublinear_tf=True
        )
        Xtr = vec.fit_transform(arr[tr])
        Xte = vec.transform(arr[te])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        accs.append(clf.score(Xte, y[te]))

    # Whole-data fit for interpretable coefficients.
    vec = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=3, sublinear_tf=True
    )
    X = vec.fit_transform(docs)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    coefs = clf.coef_[0]
    vocab = np.array(vec.get_feature_names_out())
    order = np.argsort(coefs)
    top_neg = [
        (str(vocab[i]), float(coefs[i])) for i in order[:top_k]
    ]  # most predictive of label_neg
    top_pos = [
        (str(vocab[i]), float(coefs[i])) for i in order[::-1][:top_k]
    ]  # most predictive of label_pos
    return {
        "label_pos": label_pos,
        "label_neg": label_neg,
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs, ddof=1)),
        "n_pos": len(texts_pos),
        "n_neg": len(texts_neg),
        "top_ngrams_for_pos": top_pos,
        "top_ngrams_for_neg": top_neg,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "experiments"
    out_dir = data_dir / "phase2_metrics" / "genre_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading story texts...")
    stego = load_texts(data_dir, "stego")
    cover_c1 = load_texts(data_dir, "cover_c1")
    cover_c2 = load_texts(data_dir, "cover_c2")
    print(f"  stego={len(stego)} c1={len(cover_c1)} c2={len(cover_c2)}")

    print("Computing dark-lexicon density + VADER sentiment...")
    sia = SentimentIntensityAnalyzer()
    m_stego = per_text_metrics(stego, sia)
    m_c1 = per_text_metrics(cover_c1, sia)
    m_c2 = per_text_metrics(cover_c2, sia)

    summary_means = {}
    for name, m in [("stego", m_stego), ("c1", m_c1), ("c2", m_c2)]:
        summary_means[name] = {
            "n": int(len(m["dark"])),
            "dark_density_mean": float(np.mean(m["dark"])),
            "dark_density_std": float(np.std(m["dark"], ddof=1)),
            "vader_compound_mean": float(np.mean(m["compound"])),
            "vader_compound_std": float(np.std(m["compound"], ddof=1)),
            "vader_neg_mean": float(np.mean(m["neg"])),
            "vader_neg_std": float(np.std(m["neg"], ddof=1)),
            "vader_pos_mean": float(np.mean(m["pos"])),
        }

    print("Running contrasts (stego vs c2, stego vs c1, c1 vs c2)...")
    contrasts = {}
    for metric_key in ("dark", "compound", "neg"):
        contrasts[metric_key] = [
            contrast(m_stego[metric_key], m_c2[metric_key], "stego", "c2"),
            contrast(m_stego[metric_key], m_c1[metric_key], "stego", "c1"),
            contrast(m_c1[metric_key], m_c2[metric_key], "c1", "c2"),
        ]

    print("Extracting character names...")
    names_stego, tot_stego, distinct_stego = name_distribution(stego)
    names_c1, tot_c1, distinct_c1 = name_distribution(cover_c1)
    names_c2, tot_c2, distinct_c2 = name_distribution(cover_c2)

    name_summary = {}
    for label, counter, total, distinct, n_docs in [
        ("stego", names_stego, tot_stego, distinct_stego, len(stego)),
        ("c1", names_c1, tot_c1, distinct_c1, len(cover_c1)),
        ("c2", names_c2, tot_c2, distinct_c2, len(cover_c2)),
    ]:
        name_summary[label] = {
            "n_docs": n_docs,
            "total_name_mentions": total,
            "distinct_names": distinct,
            "mentions_per_doc": total / n_docs if n_docs else 0.0,
            "top20": counter.most_common(20),
        }

    name_overlap_vs_c2 = {
        "stego_vs_c2_top20_jaccard": jaccard_topk(names_stego, names_c2, 20),
        "stego_vs_c2_top50_jaccard": jaccard_topk(names_stego, names_c2, 50),
        "stego_vs_c1_top20_jaccard": jaccard_topk(names_stego, names_c1, 20),
        "stego_vs_c1_top50_jaccard": jaccard_topk(names_stego, names_c1, 50),
        "c1_vs_c2_top20_jaccard": jaccard_topk(names_c1, names_c2, 20),
        "c1_vs_c2_top50_jaccard": jaccard_topk(names_c1, names_c2, 50),
    }

    # Names that appear in stego but NOT in c2 top lists (and vice versa).
    top_stego = {n for n, _ in names_stego.most_common(30)}
    top_c1 = {n for n, _ in names_c1.most_common(30)}
    top_c2 = {n for n, _ in names_c2.most_common(30)}
    name_set_diffs = {
        "in_stego_top30_not_c2_top30": sorted(top_stego - top_c2),
        "in_c2_top30_not_stego_top30": sorted(top_c2 - top_stego),
        "in_stego_top30_and_c2_top30": sorted(top_stego & top_c2),
        "in_stego_top30_and_c1_top30": sorted(top_stego & top_c1),
    }

    print("Fitting TF-IDF discriminator stego vs c2...")
    discrim_stego_vs_c2 = tfidf_discriminators(
        texts_pos=stego,
        texts_neg=cover_c2,
        label_pos="stego",
        label_neg="c2 (qwen outline + gpt-4.1)",
        top_k=25,
    )
    print("Fitting TF-IDF discriminator stego vs c1...")
    discrim_stego_vs_c1 = tfidf_discriminators(
        texts_pos=stego,
        texts_neg=cover_c1,
        label_pos="stego",
        label_neg="c1",
        top_k=25,
    )

    result = {
        "class_means": summary_means,
        "contrasts": contrasts,
        "name_summary": name_summary,
        "name_overlap": name_overlap_vs_c2,
        "name_set_diffs": name_set_diffs,
        "tfidf_discriminators": {
            "stego_vs_c2": discrim_stego_vs_c2,
            "stego_vs_c1": discrim_stego_vs_c1,
        },
        "lexicon_size": len(DARK_LEXICON),
    }

    out_json = out_dir / "story_genre_diagnostic.json"
    with out_json.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote {out_json}")

    md_lines = ["# Story genre diagnostic (Phase 2d)", ""]
    md_lines.append("## Per-class means (n=300 each)")
    md_lines.append("")
    md_lines.append(
        "| class | dark density | VADER compound | VADER neg | VADER pos |"
    )
    md_lines.append("|---|---:|---:|---:|---:|")
    for cls in ["stego", "c1", "c2"]:
        s = summary_means[cls]
        md_lines.append(
            f"| {cls} | {s['dark_density_mean']*100:.2f}% (±{s['dark_density_std']*100:.2f}) "
            f"| {s['vader_compound_mean']:+.3f} (±{s['vader_compound_std']:.3f}) "
            f"| {s['vader_neg_mean']:.3f} (±{s['vader_neg_std']:.3f}) "
            f"| {s['vader_pos_mean']:.3f} |"
        )
    md_lines += ["", "## Contrasts (Welch t / Mann-Whitney / Cohen's d)", ""]
    metric_labels = {
        "dark": "Dark-lexicon density",
        "compound": "VADER compound",
        "neg": "VADER neg",
    }
    for mk, mlabel in metric_labels.items():
        md_lines.append(f"### {mlabel}")
        md_lines.append("")
        md_lines.append("| contrast | mean_a | mean_b | Welch p | MW p | Cohen's d |")
        md_lines.append("|---|---:|---:|---:|---:|---:|")
        for c in contrasts[mk]:
            md_lines.append(
                f"| {c['contrast']} | {c['mean_a']:.4f} | {c['mean_b']:.4f} | "
                f"{c['welch_p']:.2e} | {c['mannwhitney_p']:.2e} | {c['cohens_d']:+.3f} |"
            )
        md_lines.append("")

    md_lines.append("## Character names")
    md_lines.append("")
    md_lines.append(
        "| class | mentions/doc | distinct names | top 10 |"
    )
    md_lines.append("|---|---:|---:|---|")
    for cls in ["stego", "c1", "c2"]:
        s = name_summary[cls]
        top10 = ", ".join(f"{n}×{c}" for n, c in s["top20"][:10])
        md_lines.append(
            f"| {cls} | {s['mentions_per_doc']:.2f} | {s['distinct_names']} | {top10} |"
        )
    md_lines += [
        "",
        f"- stego vs c2 (Jaccard top-20): {name_overlap_vs_c2['stego_vs_c2_top20_jaccard']:.3f}",
        f"- stego vs c1 (Jaccard top-20): {name_overlap_vs_c2['stego_vs_c1_top20_jaccard']:.3f}",
        f"- c1 vs c2    (Jaccard top-20): {name_overlap_vs_c2['c1_vs_c2_top20_jaccard']:.3f}",
        "",
        "Names in stego top-30 but not c2 top-30: "
        + ", ".join(name_set_diffs["in_stego_top30_not_c2_top30"]),
        "",
        "Names in c2 top-30 but not stego top-30: "
        + ", ".join(name_set_diffs["in_c2_top30_not_stego_top30"]),
        "",
    ]

    md_lines += [
        "## TF-IDF discriminators",
        "",
        f"stego vs c2 — 5-fold CV accuracy: "
        f"{discrim_stego_vs_c2['cv_accuracy_mean']:.3f} "
        f"(±{discrim_stego_vs_c2['cv_accuracy_std']:.3f})",
        "",
        "Top 15 n-grams predictive of stego:",
    ]
    for ng, w in discrim_stego_vs_c2["top_ngrams_for_pos"][:15]:
        md_lines.append(f"- `{ng}`  ({w:+.3f})")
    md_lines += ["", "Top 15 n-grams predictive of c2 (Qwen outline + GPT-4.1):"]
    for ng, w in discrim_stego_vs_c2["top_ngrams_for_neg"][:15]:
        md_lines.append(f"- `{ng}`  ({w:+.3f})")

    out_md = out_dir / "story_genre_diagnostic.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
