"""
Dataset Generation for Summary-Based Steganography PCA Training
Using CNN/DailyMail Dataset

Generates semantically-anchored facts and summary-style paraphrases from
real CNN news articles to train PCA components for stable hashing.

Requirements:
    pip install datasets
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm
from datasets import load_dataset

API_BASE = "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"

# Generation settings
NUM_ARTICLES = 100  # Number of CNN articles to process
FACTS_PER_ARTICLE = 20  # Facts to extract per article
PARAPHRASES_PER_FACT = 2  # Summary-style variations per fact
MIN_ARTICLE_LENGTH = 500  # Minimum article length in characters
MAX_ARTICLE_LENGTH = 5000  # Maximum article length in characters

# Output paths
OUT_DIR = Path("./pca/summary/artifacts/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARTICLES_JSON = OUT_DIR / "cnn_articles.json"
FACTS_JSON = OUT_DIR / "extracted_facts.json"
PARAPHRASE_JSON = OUT_DIR / "fact_paraphrases.json"
FACT_EMBED_NPY = OUT_DIR / "fact_embeddings.npy"
PARAPHRASE_EMBED_NPY = OUT_DIR / "paraphrase_embeddings.npy"
GROUP_LABELS_NPY = OUT_DIR / "paraphrase_group_labels.npy"
IMPORTANCE_LABELS_NPY = OUT_DIR / "importance_labels.npy"
STABILITY_ANALYSIS_JSON = OUT_DIR / "stability_analysis.json"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

#### Prompts ####

# Fact extraction with semantic anchoring
FACT_EXTRACTION_ANCHORED = """Extract exactly {k} facts from this news article.

Article:
{article}

REQUIREMENTS FOR EACH FACT:
1. Must be a single, complete declarative sentence
2. Must be DIRECTLY extractable from the article (use exact wording where possible)
3. Must contain AT LEAST TWO of these semantic anchors:
   - A specific named entity (full name of person, organization, or place)
   - A specific number, quantity, or measurement
   - A specific date, time, or temporal reference
   - A concrete action verb (not generic verbs like "is", "was", "said")

STRICT RULES:
- Extract using the article's exact wording where possible
- Do NOT paraphrase or summarize - pull directly from source
- Do NOT combine information from separate parts of the article
- Each fact must be independently verifiable from article text

Order facts by importance (most central to the story first).

Output format:
1. [Fact with anchors]
2. [Fact with anchors]
...
{k}. [Fact with anchors]

Output only the numbered list."""

# Summary-style paraphrase (simulates sentence in a summary)
SUMMARY_STYLE_PARAPHRASE = """Rewrite this fact as it would appear in a news summary paragraph.

Original: {fact}

Requirements:
- Preserve ALL specific details: names, numbers, dates, locations
- Change sentence structure (active↔passive, clause reordering)
- May add a brief transitional phrase at the start
- Use different word choices where possible WITHOUT changing meaning
- Keep as a single sentence
- Maintain journalistic tone

The core factual content and all anchors (names, numbers, dates) must remain intact.

Output only the rewritten sentence."""

# Alternative paraphrase for variety
SUMMARY_STYLE_PARAPHRASE_ALT = """Transform this fact for inclusion in a summary article.

Original: {fact}

Transform by:
- Restructuring the sentence (different clause order)
- Using synonyms for non-anchor words
- Optionally adding context phrase (e.g., "According to reports,")

PRESERVE EXACTLY:
- All proper nouns and names
- All numbers and quantities  
- All dates and times
- The core action/event

Output only the transformed sentence."""


def chat_completion(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 500,
    max_retries: int = 3,
) -> str:
    """Make a chat completion request with retries."""
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{API_BASE}/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=90,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Retry {attempt + 1}: {e}")
            time.sleep(2**attempt)
    return ""


def embed_texts_chunked(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Embed texts in batches."""
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        payload = {"model": EMBED_MODEL, "input": batch}

        for attempt in range(3):
            try:
                r = requests.post(
                    f"{API_BASE}/embeddings",
                    headers=HEADERS,
                    json=payload,
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()["data"]
                all_embeddings.extend([d["embedding"] for d in data])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"Embed retry {attempt + 1}: {e}")
                time.sleep(2**attempt)

        time.sleep(0.2)

    return np.array(all_embeddings)


def load_or_init_json(path: Path) -> list[dict]:
    """Load existing JSON or return empty list."""
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {path.name}")
        return data
    return []


def save_json(data: list[dict], path: Path) -> None:
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_cnn_articles() -> list[dict]:
    """Load and filter articles from CNN/DailyMail dataset."""
    articles = load_or_init_json(ARTICLES_JSON)

    if len(articles) >= NUM_ARTICLES:
        print(f"Already have {len(articles)} articles cached.")
        return articles[:NUM_ARTICLES]

    existing_ids = {a["article_id"] for a in articles}

    print("Loading CNN/DailyMail dataset from HuggingFace...")
    dataset = load_dataset(
        "cnn_dailymail", "3.0.0", split="train", trust_remote_code=True
    )

    print(f"Dataset loaded with {len(dataset)} articles. Filtering...")

    # Filter and select articles
    candidates = []
    for idx, item in enumerate(dataset):
        article_text = item["article"]

        # Filter by length
        if len(article_text) < MIN_ARTICLE_LENGTH:
            continue
        if len(article_text) > MAX_ARTICLE_LENGTH:
            continue

        # Skip if already processed
        if idx in existing_ids:
            continue

        candidates.append(
            {
                "article_id": idx,
                "text": article_text,
                "highlights": item["highlights"],  # Store reference summary
                "source": "cnn_dailymail",
            }
        )

        if (
            len(candidates) + len(articles) >= NUM_ARTICLES * 2
        ):  # Get extra in case some fail
            break

    print(f"Found {len(candidates)} candidate articles after filtering.")

    # Add new articles up to target
    needed = NUM_ARTICLES - len(articles)
    articles.extend(candidates[:needed])

    save_json(articles, ARTICLES_JSON)
    print(f"Saved {len(articles)} articles to {ARTICLES_JSON.name}")

    return articles[:NUM_ARTICLES]


def extract_facts(articles: list[dict]) -> list[dict]:
    """Extract semantically-anchored facts from each article."""
    facts = load_or_init_json(FACTS_JSON)
    existing_keys = {(f["article_id"], f["fact_index"]) for f in facts}

    for article in tqdm(articles, desc="Extracting facts"):
        article_id = article["article_id"]

        # Check if all facts exist
        existing_count = sum(1 for f in facts if f["article_id"] == article_id)
        if existing_count >= FACTS_PER_ARTICLE:
            continue

        messages = [
            {
                "role": "user",
                "content": FACT_EXTRACTION_ANCHORED.format(
                    k=FACTS_PER_ARTICLE,
                    article=article["text"],
                ),
            }
        ]

        try:
            response = chat_completion(messages, temperature=0.2, max_tokens=1500)
        except Exception as e:
            print(f"Error extracting facts for article {article_id}: {e}")
            continue

        # Parse numbered facts
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            for i in range(1, FACTS_PER_ARTICLE + 1):
                prefix = f"{i}."
                if line.startswith(prefix):
                    fact_text = line[len(prefix) :].strip()
                    if fact_text and (article_id, i - 1) not in existing_keys:
                        facts.append(
                            {
                                "article_id": article_id,
                                "fact_index": i - 1,
                                "importance_rank": i - 1,
                                "text": fact_text,
                            }
                        )
                        existing_keys.add((article_id, i - 1))
                    break

        save_json(facts, FACTS_JSON)
        time.sleep(0.2)

    return facts


def generate_paraphrases(facts: list[dict]) -> list[dict]:
    """Generate summary-style paraphrases for each fact."""
    paraphrases = load_or_init_json(PARAPHRASE_JSON)
    existing_keys = {
        (p["article_id"], p["fact_index"], p["paraphrase_index"]) for p in paraphrases
    }

    prompts = [SUMMARY_STYLE_PARAPHRASE, SUMMARY_STYLE_PARAPHRASE_ALT]

    for fact in tqdm(facts, desc="Generating paraphrases"):
        article_id = fact["article_id"]
        fact_index = fact["fact_index"]
        original_text = fact["text"]

        for para_idx in range(PARAPHRASES_PER_FACT):
            if (article_id, fact_index, para_idx) in existing_keys:
                continue

            # Alternate prompts for variety
            prompt_template = prompts[para_idx % len(prompts)]

            messages = [
                {
                    "role": "user",
                    "content": prompt_template.format(fact=original_text),
                }
            ]

            try:
                paraphrased = chat_completion(messages, temperature=0.7)
                paraphrased = paraphrased.strip().strip('"')
            except Exception as e:
                print(f"Error paraphrasing fact {article_id}-{fact_index}: {e}")
                continue

            if paraphrased:
                paraphrases.append(
                    {
                        "article_id": article_id,
                        "fact_index": fact_index,
                        "paraphrase_index": para_idx,
                        "original_text": original_text,
                        "paraphrased_text": paraphrased,
                        "importance_rank": fact["importance_rank"],
                    }
                )
                existing_keys.add((article_id, fact_index, para_idx))

            time.sleep(0.15)

        # Save periodically
        if len(paraphrases) % 50 == 0:
            save_json(paraphrases, PARAPHRASE_JSON)

    save_json(paraphrases, PARAPHRASE_JSON)
    return paraphrases


def compute_embeddings(facts: list[dict], paraphrases: list[dict]) -> None:
    """Compute and save embeddings for facts and paraphrases."""
    # Fact embeddings
    fact_texts = [f["text"] for f in facts]
    if FACT_EMBED_NPY.exists():
        fact_emb = np.load(FACT_EMBED_NPY)
        if len(fact_emb) < len(fact_texts):
            print(f"Computing {len(fact_texts) - len(fact_emb)} new fact embeddings...")
            new_emb = embed_texts_chunked(fact_texts[len(fact_emb) :])
            fact_emb = np.vstack([fact_emb, new_emb])
            np.save(FACT_EMBED_NPY, fact_emb)
    else:
        print(f"Computing {len(fact_texts)} fact embeddings...")
        fact_emb = embed_texts_chunked(fact_texts)
        np.save(FACT_EMBED_NPY, fact_emb)
    print(f"Fact embeddings: {fact_emb.shape}")

    if paraphrases:
        para_texts = [p["paraphrased_text"] for p in paraphrases]
        if PARAPHRASE_EMBED_NPY.exists():
            para_emb = np.load(PARAPHRASE_EMBED_NPY)
            if len(para_emb) < len(para_texts):
                print(
                    f"Computing {len(para_texts) - len(para_emb)} new paraphrase embeddings..."
                )
                new_emb = embed_texts_chunked(para_texts[len(para_emb) :])
                para_emb = np.vstack([para_emb, new_emb])
                np.save(PARAPHRASE_EMBED_NPY, para_emb)
        else:
            print(f"Computing {len(para_texts)} paraphrase embeddings...")
            para_emb = embed_texts_chunked(para_texts)
            np.save(PARAPHRASE_EMBED_NPY, para_emb)
        print(f"Paraphrase embeddings: {para_emb.shape}")

        # Group labels
        fact_to_idx = {
            (f["article_id"], f["fact_index"]): i for i, f in enumerate(facts)
        }
        group_labels = np.array(
            [fact_to_idx[(p["article_id"], p["fact_index"])] for p in paraphrases]
        )
        np.save(GROUP_LABELS_NPY, group_labels)
        print(f"Group labels: {group_labels.shape}")

        # Importance labels
        importance_labels = np.array([f["importance_rank"] for f in facts])
        np.save(IMPORTANCE_LABELS_NPY, importance_labels)
        print(f"Importance labels: {importance_labels.shape}")


def analyze_stability(facts: list[dict], paraphrases: list[dict]) -> dict:
    """Analyze embedding stability for PCA training insights."""
    if not FACT_EMBED_NPY.exists() or not PARAPHRASE_EMBED_NPY.exists():
        print("Embeddings not found. Run compute_embeddings first.")
        return {}

    fact_emb = np.load(FACT_EMBED_NPY)
    para_emb = np.load(PARAPHRASE_EMBED_NPY)

    fact_to_idx = {(f["article_id"], f["fact_index"]): i for i, f in enumerate(facts)}

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Compute similarities
    within_sims = []
    between_sims = []

    for i, p in enumerate(paraphrases):
        fact_idx = fact_to_idx[(p["article_id"], p["fact_index"])]

        # Within-group
        sim = cosine_sim(fact_emb[fact_idx], para_emb[i])
        within_sims.append(sim)

        # Between-group (random different fact)
        other_idx = (fact_idx + 7) % len(fact_emb)  # Offset to get different fact
        sim_other = cosine_sim(para_emb[i], fact_emb[other_idx])
        between_sims.append(sim_other)

    # Per-fact stability
    fact_stabilities = []
    for fact_idx in range(len(facts)):
        para_indices = [
            i
            for i, p in enumerate(paraphrases)
            if fact_to_idx.get((p["article_id"], p["fact_index"])) == fact_idx
        ]
        if para_indices:
            sims = [cosine_sim(fact_emb[fact_idx], para_emb[pi]) for pi in para_indices]
            fact_stabilities.append(
                {
                    "fact_idx": fact_idx,
                    "text": facts[fact_idx]["text"],
                    "importance_rank": facts[fact_idx]["importance_rank"],
                    "mean_similarity": float(np.mean(sims)),
                    "min_similarity": float(np.min(sims)),
                    "std_similarity": float(np.std(sims)),
                }
            )

    fact_stabilities.sort(key=lambda x: x["mean_similarity"], reverse=True)

    analysis = {
        "dataset": "cnn_dailymail",
        "overall": {
            "total_facts": len(facts),
            "total_paraphrases": len(paraphrases),
            "within_group_mean": float(np.mean(within_sims)),
            "within_group_std": float(np.std(within_sims)),
            "within_group_min": float(np.min(within_sims)),
            "between_group_mean": float(np.mean(between_sims)),
            "between_group_std": float(np.std(between_sims)),
            "separation_ratio": float(np.mean(within_sims) / np.mean(between_sims)),
        },
        "most_stable_facts": fact_stabilities[:10],
        "least_stable_facts": fact_stabilities[-10:],
        "stability_by_importance": {},
    }

    # Analyze stability by importance rank
    for rank in range(FACTS_PER_ARTICLE):
        rank_facts = [f for f in fact_stabilities if f["importance_rank"] == rank]
        if rank_facts:
            analysis["stability_by_importance"][f"rank_{rank}"] = {
                "count": len(rank_facts),
                "mean_stability": float(
                    np.mean([f["mean_similarity"] for f in rank_facts])
                ),
            }

    save_json(analysis, STABILITY_ANALYSIS_JSON)
    return analysis


def print_stability_report(analysis: dict) -> None:
    """Print formatted stability analysis."""
    print("\n" + "=" * 70)
    print("EMBEDDING STABILITY ANALYSIS FOR PCA TRAINING")
    print(f"Dataset: {analysis.get('dataset', 'unknown')}")
    print("=" * 70)

    overall = analysis.get("overall", {})
    print("\nOverall Statistics:")
    print(f"   Facts: {overall.get('total_facts', 0)}")
    print(f"   Paraphrases: {overall.get('total_paraphrases', 0)}")
    print("\nWithin-group similarity (original↔paraphrase):")
    print(f"      Mean: {overall.get('within_group_mean', 0):.4f}")
    print(f"      Std:  {overall.get('within_group_std', 0):.4f}")
    print(f"      Min:  {overall.get('within_group_min', 0):.4f}")
    print("\nBetween-group similarity (paraphrase↔different fact):")
    print(f"      Mean: {overall.get('between_group_mean', 0):.4f}")
    print(f"\nSeparation ratio: {overall.get('separation_ratio', 0):.4f}")
    print("   (Higher = better for hashing stability)")

    print("\nMost Stable Facts (best for encoding):")
    for f in analysis.get("most_stable_facts", [])[:5]:
        print(f"   [{f['mean_similarity']:.4f}] {f['text'][:70]}...")

    print("\nLeast Stable Facts (risky for encoding):")
    for f in analysis.get("least_stable_facts", [])[:5]:
        print(f"   [{f['mean_similarity']:.4f}] {f['text'][:70]}...")

    print("\nStability by Importance Rank:")
    for rank, stats in sorted(analysis.get("stability_by_importance", {}).items()):
        print(f"   {rank}: {stats['mean_stability']:.4f} (n={stats['count']})")


def main():
    print("=" * 70)
    print("Summary System Dataset Generation for PCA Training")
    print("Using CNN/DailyMail Dataset")
    print("=" * 70)
    print(
        f"Config: {NUM_ARTICLES} articles × {FACTS_PER_ARTICLE} facts × {PARAPHRASES_PER_FACT} paraphrases"
    )

    print("\n[1/5] Loading CNN/DailyMail articles...")
    articles = load_cnn_articles()
    print(f"Total articles: {len(articles)}")

    print("\n[2/5] Extracting anchored facts...")
    facts = extract_facts(articles)
    print(f"Total facts: {len(facts)}")

    print("\n[3/5] Generating summary-style paraphrases...")
    paraphrases = generate_paraphrases(facts)
    print(f"Total paraphrases: {len(paraphrases)}")

    print("\n[4/5] Computing embeddings...")
    compute_embeddings(facts, paraphrases)

    print("\n[5/5] Analyzing stability...")
    analysis = analyze_stability(facts, paraphrases)
    print_stability_report(analysis)

    print("\n" + "=" * 70)
    print("Dataset generation complete!")
    print(f"Artifacts saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
