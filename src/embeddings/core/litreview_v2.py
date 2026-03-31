import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config.litreview_prompts import EXTRACT_CITATIONS, GENERATE_REVIEW
from .encoder import CharacterEncoder, Encoder
from .error_correction import ErrorCorrection
from .steg_system import StegSystem


def _llm(
    client,
    model,
    prompt,
    system="You are a helpful assistant.",
    temperature=0,
    max_tokens=1000,
):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(2**attempt)


def ref_bit_hash(author_last_name: str, year: int) -> int:
    key = f"{author_last_name.lower().strip()}_{year}"
    return hashlib.sha256(key.encode()).digest()[0] % 2


def load_corpus(corpus_path: str | Path, refs_path: str | Path) -> list[dict]:
    """Load and join corpus.jsonl + references.jsonl into a list of paper dicts.

    Each entry has: paperId, title, abstract, year, authors, references.
    Only papers present in both files and with non-empty references are included.
    Returns a deterministically sorted list (by paperId).
    """
    corpus_path, refs_path = Path(corpus_path), Path(refs_path)

    papers: dict[str, dict] = {}
    with open(corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            papers[obj["paperId"]] = obj

    with open(refs_path) as f:
        for line in f:
            obj = json.loads(line)
            pid = obj["paperId"]
            if pid in papers:
                papers[pid]["references"] = obj["references"]

    joined = [p for p in papers.values() if p.get("references") and p.get("abstract")]
    joined.sort(key=lambda p: p["paperId"])
    return joined


def prepare_references(raw_refs: list[dict], min_title_words: int = 5) -> list[dict]:
    """Convert raw reference dicts from references.jsonl into sorted, deduplicated
    refs with hash_bit. Filters out refs with short titles."""
    refs = []
    seen_keys: set[str] = set()

    for r in raw_refs:
        last_name = r.get("author_last_name", "")
        year = r.get("year")
        title = r.get("title", "")
        if not last_name or not year or not title:
            continue
        if len(title.split()) < min_title_words:
            continue

        dedup_key = f"{last_name.lower()}_{year}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        refs.append(
            {
                "year": year,
                "author_text": r.get("author_text", last_name),
                "author_last_name": last_name,
                "ref_title": title,
                "hash_bit": ref_bit_hash(last_name, year),
            }
        )

    refs.sort(key=lambda r: (r["year"], r["author_last_name"].lower()))
    return refs


def extract_citations(client, model, text: str) -> list[dict]:
    response = _llm(
        client,
        model,
        prompt=text,
        system=EXTRACT_CITATIONS,
        temperature=0,
        max_tokens=2000,
    )

    seen: set[str] = set()
    citations = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[.):\s]+", "", line).strip()
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue
        last_name, year_str = parts
        last_name = last_name.strip(".,;:()")
        if not re.match(r"\d{4}$", year_str):
            continue

        year = int(year_str)
        dedup_key = f"{last_name.lower()}_{year}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        citations.append(
            {
                "author_last_name": last_name,
                "year": year,
                "hash_bit": ref_bit_hash(last_name, year),
            }
        )

    citations.sort(key=lambda c: (c["year"], c["author_last_name"].lower()))
    return citations


def greedy_select(references: list[dict], message_bits: list[int]) -> list[dict] | None:
    selected: list[dict] = []
    ref_idx = 0
    for bit in message_bits:
        while ref_idx < len(references):
            if references[ref_idx]["hash_bit"] == bit:
                selected.append(references[ref_idx])
                ref_idx += 1
                break
            ref_idx += 1
        else:
            return None
    return selected


class LitReviewSystemV2(StegSystem):
    def __init__(
        self,
        client,
        error_correction: ErrorCorrection,
        corpus: list[dict] | None = None,
        model: str = "gpt-4.1",
        encoder: Encoder | None = None,
    ):
        self.client = client
        self.model = model
        self.hash_fn = None
        self.ecc = error_correction
        self.encoder = encoder or CharacterEncoder()
        self.hash_output_length = 1
        self._error_encoded_length: int | None = None
        self.corpus = corpus
        if not self.corpus:
            self.corpus = load_corpus(
                "src/pca/litreview/references/corpus.jsonl",
                "src/pca/litreview/references/references.jsonl",
            )

    def encode(self, chunks, history, system_prompt, max_length=200, **kwargs):
        raise NotImplementedError(
            "LitReviewSystem encodes via reference selection. Use hide_message directly."
        )

    def hide_message(self, data: Any, seed: str, **kwargs) -> str:
        """Encode data into a literature review for the paper at corpus index `seed`."""
        if self.corpus is None:
            raise ValueError("Corpus not initialized")
        paper = self.corpus[int(seed)]
        references = prepare_references(paper["references"])

        chunks, self._error_encoded_length = self._encode_to_chunks(data)
        message_bits = [int(c[0]) for c in chunks]

        n_zeros = sum(1 for r in references if r["hash_bit"] == 0)
        n_ones = len(references) - n_zeros
        print(
            f"  Pool: {len(references)} refs (0s: {n_zeros}, 1s: {n_ones}), encoding {len(message_bits)} bits"
        )

        selected = greedy_select(references, message_bits)
        if selected is None:
            raise ValueError(
                f"Reference pool exhausted: cannot encode "
                f"{''.join(map(str, message_bits))} with "
                f"{len(references)} references (0s: {n_zeros}, 1s: {n_ones})"
            )

        print(
            f"  Selected {len(selected)} references, bits: {''.join(str(r['hash_bit']) for r in selected)}"
        )

        stego_text = self._generate_review(paper, selected)
        return stego_text

    def recover_message(self, stego_text: str, **kwargs):
        if self._error_encoded_length is None:
            raise ValueError(
                "No encoded length set. Run hide_message first or set error_encoded_length."
            )
        expected_bits = self._error_encoded_length // self.hash_output_length

        citations = extract_citations(self.client, self.model, stego_text)
        recovered_bits = [c["hash_bit"] for c in citations]
        print(
            f"  Extracted {len(citations)} citations, bits: {''.join(map(str, recovered_bits))}"
        )

        if len(recovered_bits) != expected_bits:
            print(
                f"  WARNING: Expected {expected_bits} bits, got {len(recovered_bits)}"
            )

        if len(recovered_bits) < expected_bits:
            recovered_bits.extend([0] * (expected_bits - len(recovered_bits)))
        recovered_bits = recovered_bits[:expected_bits]

        ecc_bits = np.array(recovered_bits)
        decoded_bits = self.ecc.decode(ecc_bits, self._error_encoded_length)
        return self.encoder.decode(decoded_bits)

    def _generate_review(self, paper: dict, selected_refs: list[dict]) -> str:
        refs_formatted = "\n".join(
            f"  - {r['author_text']} ({r['year']}). {r['ref_title']}"
            for r in selected_refs
        )

        return _llm(
            self.client,
            self.model,
            prompt=f"References:\n{refs_formatted}",
            system=GENERATE_REVIEW.format(
                seed_title=paper["title"],
                seed_abstract=paper.get("abstract", "")[:600],
            ),
            temperature=0,
            max_tokens=4000,
        )
