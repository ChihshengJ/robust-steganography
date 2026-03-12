import hashlib
import json
import re
from typing import Any

import numpy as np

from ..config.litreview_prompts import EXTRACT_CITATIONS, GENERATE_REVIEW
from ..utils.new_text import generate_response
from .encoder import Encoder, CharacterEncoder
from .error_correction import ErrorCorrection
from .steg_system import StegSystem


def ref_bit_hash(author_last_name: str, year: int) -> int:
    key = f"{author_last_name.lower().strip()}_{year}"
    return hashlib.sha256(key.encode()).digest()[0] % 2


def normalize_year(year_str) -> int:
    if year_str is None:
        return 0
    digits = re.match(r"(\d{4})", str(year_str))
    return int(digits.group(1)) if digits else 0


def parse_cached_text(cached_text: str) -> dict:
    text = cached_text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^\[\d+\]\s*", "", text).strip()

    parts = re.split(r",|\band\b", text, maxsplit=1)
    author_token = parts[0].strip()

    et_al = re.search(r"et\s+al\.?", text)
    author_text = text[: et_al.end()].strip() if et_al else author_token

    author_last_name = author_token.split()[-1] if author_token else ""
    author_last_name = author_last_name.strip(".,;:()")

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

    return {
        "author_text": author_text,
        "author_last_name": author_last_name,
        "ref_title": ref_title.strip().rstrip("."),
    }


def parse_references(anchors: list[dict], min_title_words: int = 5) -> list[dict]:
    refs = []
    seen_keys: set[str] = set()

    for anchor in anchors:
        cached = anchor.get("cachedText", "")
        year_raw = anchor.get("year")
        if not cached or not year_raw:
            continue
        year = normalize_year(year_raw)
        if year == 0:
            continue

        parsed = parse_cached_text(cached)
        if len(parsed["ref_title"].split()) < min_title_words:
            continue
        if not all(
            ord(c) < 128 or c in "\u2018\u2019\u2013\u2014"
            for c in parsed["ref_title"]
        ):
            continue

        dedup_key = f"{parsed['author_last_name'].lower()}_{year}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        refs.append(
            {
                "year": year,
                "author_text": parsed["author_text"],
                "author_last_name": parsed["author_last_name"],
                "ref_title": parsed["ref_title"],
                "hash_bit": ref_bit_hash(parsed["author_last_name"], year),
            }
        )

    refs.sort(key=lambda r: (r["year"], r["author_last_name"].lower()))
    return refs


def extract_citations(text: str) -> list[dict]:
    response = generate_response(
        prompt=text,
        system_prompt=EXTRACT_CITATIONS,
        max_length=2000,
        temperature=0,
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


def greedy_select(
    references: list[dict], message_bits: list[int]
) -> list[dict] | None:
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
        encoder: Encoder | None = None,
    ):
        self.client = client
        self.hash_fn = None
        self.ecc = error_correction
        self.encoder = encoder or CharacterEncoder()
        self.hash_output_length = 1
        self._error_encoded_length: int | None = None

    def encode(self, chunks, history, system_prompt, max_length=200, **kwargs):
        raise NotImplementedError(
            "LitReviewSystem encodes via reference selection. Use hide_message directly."
        )

    def hide_message(self, data: Any, seed: str | dict, **kwargs) -> str:
        seed_data = json.loads(seed) if isinstance(seed, str) else seed
        references = parse_references(seed_data["anchors"])

        chunks, self._error_encoded_length = self._encode_to_chunks(data)
        message_bits = [int(c[0]) for c in chunks]

        n_zeros = sum(1 for r in references if r["hash_bit"] == 0)
        n_ones = len(references) - n_zeros
        print(f"  Pool: {len(references)} refs (0s: {n_zeros}, 1s: {n_ones}), encoding {len(message_bits)} bits")

        selected = greedy_select(references, message_bits)
        if selected is None:
            raise ValueError(
                f"Reference pool exhausted: cannot encode "
                f"{''.join(map(str, message_bits))} with "
                f"{len(references)} references (0s: {n_zeros}, 1s: {n_ones})"
            )

        print(f"  Selected {len(selected)} references, bits: {''.join(str(r['hash_bit']) for r in selected)}")

        stego_text = self._generate_review(seed_data, selected)
        return stego_text

    def recover_message(self, stego_text: str, **kwargs):
        if self._error_encoded_length is None:
            raise ValueError(
                "No encoded length set. Run hide_message first or set error_encoded_length."
            )
        expected_bits = self._error_encoded_length // self.hash_output_length

        citations = extract_citations(stego_text)
        recovered_bits = [c["hash_bit"] for c in citations]
        print(f"  Extracted {len(citations)} citations, bits: {''.join(map(str, recovered_bits))}")

        if len(recovered_bits) != expected_bits:
            print(f"  WARNING: Expected {expected_bits} bits, got {len(recovered_bits)}")

        if len(recovered_bits) < expected_bits:
            recovered_bits.extend([0] * (expected_bits - len(recovered_bits)))
        recovered_bits = recovered_bits[:expected_bits]

        ecc_bits = np.array(recovered_bits)
        decoded_bits = self.ecc.decode(ecc_bits, self._error_encoded_length)
        return self.encoder.decode(decoded_bits)

    def _generate_review(self, seed_data: dict, selected_refs: list[dict]) -> str:
        refs_formatted = "\n".join(
            f"  - {r['author_text']} ({r['year']}). {r['ref_title']}"
            for r in selected_refs
        )

        response = generate_response(
            prompt=f"References:\n{refs_formatted}",
            system_prompt=GENERATE_REVIEW.format(
                seed_title=seed_data["title"],
                seed_abstract=seed_data["abstract"][:600],
            ),
            max_length=4000,
            temperature=0,
        )
        return response
