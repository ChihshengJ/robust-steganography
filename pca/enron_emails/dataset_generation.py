import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

API_BASE = "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"

PARAPHRASES_PER_SENT = 2
MIN_CHAR_LENGTH = 60
DATASET_CUTOFF = 1500

OUT_DIR = Path("./pca/enron_emails/artifacts/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMAIL_PATH = Path("./pca/enron_emails/enron_paragraphs.json")
SENTENCES_JSON = OUT_DIR / "sentences.json"
PARAPHRASE_JSON = OUT_DIR / "paraphrase_pairs.json"
EMBED_NPY = OUT_DIR / "sentence_embeddings.npy"
PARAPHRASE_EMBED_NPY = OUT_DIR / "paraphrase_embeddings.npy"
GROUP_LABELS_NPY = OUT_DIR / "paraphrase_group_labels.npy"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

PARAPHRASE_SYSTEM_PROMPT = """You are a text paraphrasing assistant. Rewrite the given text while:

1. PRESERVING all factual information (names, places, numbers, actions)
2. CHANGING the sentence structure significantly
3. USING different vocabulary where possible
4. MAINTAINING the same meaning and tone

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
            print(f"Retry {attempt + 1}: {e}")
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
                print(f"Retry {attempt + 1}: {e}")
                time.sleep(2**attempt)

        time.sleep(0.3)

    return np.array(all_embeddings)


def save_json(data, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_sentences(path: Path) -> list[dict]:
    if SENTENCES_JSON.exists():
        with open(SENTENCES_JSON) as f:
            sentences = json.load(f)
        print(f"Loaded {len(sentences)} sentences from cache")
        return sentences

    with open(path) as f:
        data = json.load(f)

    paragraphs = data["paragraphs"][:DATASET_CUTOFF]
    sentences = []
    for para_idx, para in enumerate(paragraphs):
        for sent_idx, sent in enumerate(para["sentences"]):
            if len(sent) >= MIN_CHAR_LENGTH:
                sentences.append(
                    {
                        "flat_idx": len(sentences),
                        "para_idx": para_idx,
                        "sent_idx": sent_idx,
                        "text": sent,
                    }
                )

    save_json(sentences, SENTENCES_JSON)
    return sentences


def generate_paraphrases(sentences: list[dict]) -> list[dict]:
    if PARAPHRASE_JSON.exists():
        with open(PARAPHRASE_JSON) as f:
            paraphrases = json.load(f)
        print(f"Loaded {len(paraphrases)} paraphrases from cache")
    else:
        paraphrases = []

    existing_keys = {(p["flat_idx"], p["paraphrase_index"]) for p in paraphrases}
    new_count = 0

    for sent in tqdm(sentences, desc="Paraphrasing"):
        flat_idx = sent["flat_idx"]

        for para_idx in range(PARAPHRASES_PER_SENT):
            if (flat_idx, para_idx) in existing_keys:
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

        group_labels = np.array([p["flat_idx"] for p in paraphrases])
        np.save(GROUP_LABELS_NPY, group_labels)

        print(f"Paraphrase embeddings: {para_emb.shape}")
        print(f"Group labels: {group_labels.shape}")


def main():
    print("-" * 60)
    print("Enron Email PCA Data Generation")
    print("-" * 60)

    print("\n[1/3] Loading sentences...")
    sentences = load_sentences(EMAIL_PATH)
    print(f"Total sentences: {len(sentences)}")

    print("\n[2/3] Generating paraphrases...")
    paraphrases = generate_paraphrases(sentences)
    print(f"Total paraphrases: {len(paraphrases)}")

    print("\n[3/3] Computing embeddings...")
    compute_embeddings(sentences, paraphrases)

    print("\n" + "-" * 60)
    print(f"Data generation complete. Artifacts saved to {OUT_DIR}")
    print("-" * 60)


if __name__ == "__main__":
    main()
