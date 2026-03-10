import json
import os
import re
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

API_BASE = "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"

GENERATIONS_PER_REF = 5
PARAPHRASES_PER_SENT = 2
MIN_TITLE_WORDS = 4

PAPERS_DIR = Path("./src/pca/litreview/papers/")
OUT_DIR = Path("./src/pca/litreview/artifacts/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCES_JSON = OUT_DIR / "references.json"
SENTENCES_JSON = OUT_DIR / "sentences.json"
PARAPHRASE_JSON = OUT_DIR / "paraphrase_pairs.json"
EMBED_NPY = OUT_DIR / "sentence_embeddings.npy"
PARAPHRASE_EMBED_NPY = OUT_DIR / "paraphrase_embeddings.npy"
GROUP_LABELS_NPY = OUT_DIR / "group_labels.npy"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

LITREVIEW_SYSTEM_PROMPT = """You are an academic researcher writing a related work section. Given the context of a seed paper and a reference paper's title, write a single sentence that could naturally appear in a literature review.

The sentence MUST:
- Mention the first author's last name and publication year as a citation, e.g., "Smith et al. (2021)"
- Describe a plausible contribution of the referenced paper based on its title
- Be a single, self-contained sentence (25-50 words)
- Sound natural and academic IMPORTANT: Each time you describe the same paper, take a COMPLETELY DIFFERENT angle. You might focus on:
- What problem the paper addresses
- What method or technique it introduces
- How it relates to or improves on earlier work
- What result or finding it demonstrates
- What broader impact or shift it represents

Output ONLY the sentence. No preamble, no numbering."""


LITREVIEW_USER_TEMPLATE = """Seed paper: "{seed_title}"
Seed abstract: {seed_abstract}

Reference to describe:
- Authors: {author_text}
- Year: {year}
- Title: "{ref_title}"

Write a related work sentence for this reference:"""


PARAPHRASE_SYSTEM_PROMPT = """You are a text paraphrasing assistant. Rewrite the given text while:

1. PRESERVING all factual information (names, years, technical terms)
2. PRESERVING all author citations exactly (e.g., "Smith et al. (2021)" must remain)
3. CHANGING the sentence structure significantly
4. USING different vocabulary where possible
5. MAINTAINING the same meaning and tone

Output ONLY the paraphrased text. No commentary."""


def chat_completion(
    messages: list[dict], temperature: float = 0.7, max_retries: int = 3
) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 300,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{API_BASE}/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  Retry {attempt + 1}: {e}")
            time.sleep(2**attempt)
    return ""


def embed_texts_chunked(texts: list[str], batch_size: int = 512) -> np.ndarray:
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
                print(f"  Retry {attempt + 1}: {e}")
                time.sleep(2**attempt)
        time.sleep(0.3)
    return np.array(all_embeddings)


def save_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_cached_text(cached_text: str) -> dict:
    text = cached_text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Author: everything before first comma or " and "
    match = re.split(r",|\band\b", text, maxsplit=1)
    author_token = match[0].strip()
    author_sort_letter = author_token[0].upper() if author_token else "Z"

    # Author display text: up to "et al." or truncate
    et_al_match = re.search(r"et\s+al\.?", text)
    if et_al_match:
        author_text = text[: et_al_match.end()].strip()
    else:
        author_text = author_token

    # Title: text between author block and venue/year markers
    title_start = re.search(r"(?:et\s+al\.?\s*[.,]?\s*|\.\s+)(?=[A-Z])", text)
    remainder = text[title_start.end() :] if title_start else text

    venue_patterns = [
        r"\.\s*(?:In\s+)",
        r"\.\s*(?:arXiv)",
        r"\.\s*(?:Proceedings)",
        r"\.\s*(?:Advances\s+in)",
        r"\.\s*(?:IEEE|ACM|AAAI|ICML|NeurIPS|ICLR|CVPR|ECCV|ICCV|ACL|EMNLP|NAACL)",
        r"\.\s*(?:Journal)",
        r"\.\s*(?:Trans(?:actions)?\.?\s)",
        r",\s*\d{4}",
        r"\.\s*\d{4}",
    ]
    ref_title = remainder
    for pattern in venue_patterns:
        m = re.search(pattern, remainder)
        if m:
            candidate = remainder[: m.start()].strip().rstrip(".")
            if len(candidate.split()) >= 3:
                ref_title = candidate
                break

    ref_title = ref_title.strip().rstrip(".")

    return {
        "author_sort_letter": author_sort_letter,
        "author_text": author_text,
        "ref_title": ref_title,
    }


def normalize_year(year_str: str) -> int:
    """Handle years like '2025b' → 2025"""
    digits = re.match(r"(\d{4})", str(year_str))
    return int(digits.group(1)) if digits else 0


# ---------------------------------------------------------------------------
# Step 1: Parse references from paper JSONs
# ---------------------------------------------------------------------------
def load_all_references() -> list[dict]:
    if REFERENCES_JSON.exists():
        with open(REFERENCES_JSON) as f:
            refs = json.load(f)
        print(f"Loaded {len(refs)} references from cache")
        return refs

    paper_files = sorted(PAPERS_DIR.glob("*.json"))
    if not paper_files:
        raise FileNotFoundError(f"No JSON files found in {PAPERS_DIR}")

    all_refs = []
    for paper_file in paper_files:
        with open(paper_file) as f:
            paper = json.load(f)

        seed_title = paper.get("title", "")
        seed_abstract = paper.get("abstract", "")
        paper_id = paper_file.stem

        if not seed_title:
            print(f"  WARNING: {paper_file.name} missing 'title', skipping")
            continue

        anchors = paper.get("anchors", [])
        print(
            f"  {paper_file.name}: {len(anchors)} anchors, title='{seed_title[:60]}...'"
        )

        for anchor in anchors:
            cached = anchor.get("cachedText", "")
            year_raw = anchor.get("year", "")
            index = anchor.get("index", 0)

            if not cached or not year_raw:
                continue

            year = normalize_year(year_raw)
            if year == 0:
                continue

            parsed = parse_cached_text(cached)
            ref_title = parsed["ref_title"]

            # Filter: skip short/empty titles
            if len(ref_title.split()) < MIN_TITLE_WORDS:
                continue

            # Filter: skip non-ASCII titles (likely non-English)
            if not all(ord(c) < 128 or c in "''–—" for c in ref_title):
                continue

            all_refs.append(
                {
                    "paper_id": paper_id,
                    "seed_title": seed_title,
                    "seed_abstract": seed_abstract,
                    "ref_index": index,
                    "year": year,
                    "year_raw": year_raw,
                    "author_sort_letter": parsed["author_sort_letter"],
                    "author_text": parsed["author_text"],
                    "ref_title": ref_title,
                    "cached_text": cached,
                }
            )

    # Sort globally by (paper_id, year, author_sort_letter)
    all_refs.sort(
        key=lambda r: (r["paper_id"], r["year"], r["author_sort_letter"].lower())
    )

    # Assign flat indices
    for i, ref in enumerate(all_refs):
        ref["flat_idx"] = i

    save_json(all_refs, REFERENCES_JSON)
    print(f"\nTotal usable references: {len(all_refs)}")
    return all_refs


# ---------------------------------------------------------------------------
# Step 2: Generate related-work sentences
# ---------------------------------------------------------------------------
def generate_sentences(references: list[dict]) -> list[dict]:
    if SENTENCES_JSON.exists():
        with open(SENTENCES_JSON) as f:
            sentences = json.load(f)
        print(f"Loaded {len(sentences)} sentences from cache")
    else:
        sentences = []

    existing_keys = {(s["flat_idx"], s["gen_index"]) for s in sentences}
    new_count = 0

    for ref in tqdm(references, desc="Generating sentences"):
        flat_idx = ref["flat_idx"]

        for gen_idx in range(GENERATIONS_PER_REF):
            if (flat_idx, gen_idx) in existing_keys:
                continue

            abstract_trunc = ref["seed_abstract"][:800]

            messages = [
                {"role": "system", "content": LITREVIEW_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": LITREVIEW_USER_TEMPLATE.format(
                        seed_title=ref["seed_title"],
                        seed_abstract=abstract_trunc,
                        author_text=ref["author_text"],
                        year=ref["year"],
                        ref_title=ref["ref_title"],
                    ),
                },
            ]

            text = chat_completion(messages, temperature=1.0)
            text = text.strip().strip('"')

            if text:
                sentences.append(
                    {
                        "flat_idx": flat_idx,
                        "gen_index": gen_idx,
                        "paper_id": ref["paper_id"],
                        "ref_index": ref["ref_index"],
                        "author_sort_letter": ref["author_sort_letter"],
                        "year": ref["year"],
                        "ref_title": ref["ref_title"],
                        "text": text,
                    }
                )
                new_count += 1

            if new_count > 0 and new_count % 50 == 0:
                save_json(sentences, SENTENCES_JSON)

            time.sleep(0.15)

    save_json(sentences, SENTENCES_JSON)
    return sentences


# ---------------------------------------------------------------------------
# Step 3: Generate paraphrases
# ---------------------------------------------------------------------------
def generate_paraphrases(sentences: list[dict]) -> list[dict]:
    if PARAPHRASE_JSON.exists():
        with open(PARAPHRASE_JSON) as f:
            paraphrases = json.load(f)
        print(f"Loaded {len(paraphrases)} paraphrases from cache")
    else:
        paraphrases = []

    existing_keys = {
        (p["flat_idx"], p["gen_index"], p["paraphrase_index"]) for p in paraphrases
    }
    new_count = 0

    for sent in tqdm(sentences, desc="Paraphrasing"):
        flat_idx = sent["flat_idx"]
        gen_idx = sent["gen_index"]

        for para_idx in range(PARAPHRASES_PER_SENT):
            if (flat_idx, gen_idx, para_idx) in existing_keys:
                continue

            messages = [
                {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Paraphrase:\n\n{sent['text']}"},
            ]

            paraphrased = chat_completion(messages)
            paraphrased = paraphrased.strip().strip('"')

            if paraphrased:
                paraphrases.append(
                    {
                        "flat_idx": flat_idx,
                        "gen_index": gen_idx,
                        "paraphrase_index": para_idx,
                        "original_text": sent["text"],
                        "paraphrased_text": paraphrased,
                    }
                )
                new_count += 1

            if new_count > 0 and new_count % 100 == 0:
                save_json(paraphrases, PARAPHRASE_JSON)

            time.sleep(0.15)

    save_json(paraphrases, PARAPHRASE_JSON)
    return paraphrases


# ---------------------------------------------------------------------------
# Step 4: Compute embeddings
# ---------------------------------------------------------------------------
def compute_embeddings(sentences: list[dict], paraphrases: list[dict]):
    sent_texts = [s["text"] for s in sentences]

    if EMBED_NPY.exists():
        sent_emb = np.load(EMBED_NPY)
        if len(sent_emb) < len(sent_texts):
            print(
                f"Computing {len(sent_texts) - len(sent_emb)} new sentence embeddings..."
            )
            new_emb = embed_texts_chunked(sent_texts[len(sent_emb) :])
            sent_emb = np.vstack([sent_emb, new_emb])
            np.save(EMBED_NPY, sent_emb)
    else:
        sent_emb = embed_texts_chunked(sent_texts)
        np.save(EMBED_NPY, sent_emb)
    print(f"Sentence embeddings: {sent_emb.shape}")

    if paraphrases:
        para_texts = [p["paraphrased_text"] for p in paraphrases]

        if PARAPHRASE_EMBED_NPY.exists():
            para_emb = np.load(PARAPHRASE_EMBED_NPY)
            if len(para_emb) < len(para_texts):
                new_emb = embed_texts_chunked(para_texts[len(para_emb) :])
                para_emb = np.vstack([para_emb, new_emb])
                np.save(PARAPHRASE_EMBED_NPY, para_emb)
        else:
            para_emb = embed_texts_chunked(para_texts)
            np.save(PARAPHRASE_EMBED_NPY, para_emb)

        # Map (flat_idx, gen_index) → position in sentences array
        sent_pos = {(s["flat_idx"], s["gen_index"]): i for i, s in enumerate(sentences)}

        # Each paraphrase points to the array position of its original sentence
        group_labels = np.array(
            [sent_pos[(p["flat_idx"], p["gen_index"])] for p in paraphrases]
        )
        np.save(GROUP_LABELS_NPY, group_labels)

        print(f"Paraphrase embeddings: {para_emb.shape}")
        print(f"Group labels: {group_labels.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Literature Review PCA Data Generation")
    print("=" * 60)

    print(f"\nPapers directory: {PAPERS_DIR}")
    print(f"Output directory: {OUT_DIR}")

    print("\n[1/4] Parsing references from paper JSONs...")
    references = load_all_references()
    print(f"Total references: {len(references)}")

    n_papers = len(set(r["paper_id"] for r in references))
    print(f"From {n_papers} papers")
    print(f"Expected sentences: {len(references) * GENERATIONS_PER_REF}")
    print(
        f"Expected paraphrases: {len(references) * GENERATIONS_PER_REF * PARAPHRASES_PER_SENT}"
    )

    print("\n[2/4] Generating related-work sentences...")
    sentences = generate_sentences(references)
    print(f"Total sentences: {len(sentences)}")

    print("\n[3/4] Generating paraphrases...")
    paraphrases = generate_paraphrases(sentences)
    print(f"Total paraphrases: {len(paraphrases)}")

    print("\n[4/4] Computing embeddings...")
    compute_embeddings(sentences, paraphrases)

    print("\n" + "=" * 60)
    print("Data generation complete.")
    print(f"Artifacts saved to {OUT_DIR}")
    print("=" * 60)

    print(f"\n  References:  {len(references)}")
    print(f"  Sentences:   {len(sentences)}")
    print(f"  Paraphrases: {len(paraphrases)}")
    print(
        f"  API calls:   ~{len(sentences) + len(paraphrases)} chat + embedding batches"
    )


if __name__ == "__main__":
    main()
