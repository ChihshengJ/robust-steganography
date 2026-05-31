import json
import re
from typing import Any

import numpy as np

from systems.config.constants import BacktrackConfig
from systems.config.litreview_prompts import (
    BASE_GENERATION,
    DECOMPOSE,
    SENTENCE_CONTINUATION,
    SENTENCE_CONTINUATION_TEMPLATE,
    SYNTHESIZE,
)
from ...utils.get_embedding import get_embeddings_in_batch
from ...utils.new_text import generate_response
from ...utils.sample_utils import (
    BacktrackingEncoder,
    RejectionSampler,
    StepChoice,
    check_near_duplicate,
)

from ..encoder import Encoder
from ..error_correction import ErrorCorrection
from ..hash_functions import HashFunction
from ..steg_system import StegSystem

# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------


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

    author_last_name = author_token.split()[0] if author_token else ""
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
            ord(c) < 128 or c in "\u2018\u2019\u2013\u2014" for c in parsed["ref_title"]
        ):
            continue

        refs.append(
            {
                "year": year,
                "author_text": parsed["author_text"],
                "author_last_name": parsed["author_last_name"],
                "ref_title": parsed["ref_title"],
            }
        )

    refs.sort(key=lambda r: (r["year"], r["author_last_name"].lower()))
    return refs


# ---------------------------------------------------------------------------
# Prompt builder, history updater, response cleaner
# ---------------------------------------------------------------------------


def build_prompt(
    history: dict, system_prompt: str, covered: set[str]
) -> tuple[str, str]:
    references = history["references"]
    base_sentences = history["base_sentences"]
    optional_sentences = history["optional_sentences"]
    all_sentences = base_sentences + optional_sentences

    ref_idx = history["key"] + len(optional_sentences)
    ref = references[ref_idx]

    prev_lines = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(all_sentences))

    prompt = SENTENCE_CONTINUATION_TEMPLATE.format(
        seed_title=history["seed_title"],
        seed_abstract=history["seed_abstract"][:600],
        author_text=ref["author_text"],
        year=ref["year"],
        ref_title=ref["ref_title"],
        previous_sentences=prev_lines or "(none yet)",
        n_previous=len(all_sentences),
    )

    sys = system_prompt
    if covered:
        recent = list(covered)[-5:]
        sys += (
            "\n\nPrevious attempts for this reference (write something DIFFERENT):\n"
            + "\n".join(f"- {s[:120]}" for s in recent)
        )
    return sys, prompt


def update_history(initial_history: dict, steps: list[StepChoice]) -> dict:
    return {
        **initial_history,
        "optional_sentences": [step.message for step in steps],
    }


def clean_response(response: str, history: dict) -> str:
    response = response.strip()
    response = re.sub(r"^\d+[.):\s]+", "", response).strip()
    response = re.sub(r"^[-\u2022*]\s*", "", response).strip()
    return response.strip("\"'")


class LitReviewSystem(StegSystem):
    def __init__(
        self,
        client,
        key: int,
        hash_function: HashFunction,
        error_correction: ErrorCorrection,
        encoder: Encoder | None = None,
        backtrack_config: BacktrackConfig | None = None,
    ):
        if key < 0:
            raise ValueError("key must be non-negative")
        super().__init__(client, hash_function, error_correction, encoder)
        self.key = key
        self.message_length: int | None = None
        self.hash_output_length = getattr(hash_function, "output_length")
        self.backtrack_config = backtrack_config or BacktrackConfig()

        sampler = RejectionSampler(
            prompt_builder=build_prompt,
            duplicate_checker=check_near_duplicate,
            response_cleaner=clean_response,
            count_duplicates=False,
        )
        self._encoder = BacktrackingEncoder(
            sampler=sampler,
            config=self.backtrack_config,
            history_updater=update_history,
        )

    def encode(
        self,
        chunks: list[list[int]],
        history: dict,
        system_prompt: str,
        max_length: int = 200,
        temperature: float = 1.0,
        **kwargs,
    ) -> tuple[list[str], list]:
        return self._encoder.encode(
            client=self.client,
            chunks=[np.array(c) for c in chunks],
            initial_history=history,
            hash_fn=self.hash_fn,
            system_prompt=system_prompt,
            max_length=max_length,
            temperature=temperature,
        )

    def hide_message(self, data: Any, seed: str | dict, **kwargs) -> str:
        seed_data = json.loads(seed) if isinstance(seed, str) else seed
        references = parse_references(seed_data["anchors"])
        chunks, self.message_length = self._encode_to_chunks(data)
        expected_optional = len(chunks)

        if len(references) < self.key + expected_optional:
            raise ValueError(
                f"Need at least {self.key + expected_optional} usable references, "
                f"got {len(references)}"
            )

        print(f"\n{'=' * 60}")
        print(
            f"Parsed {len(references)} usable references from {len(seed_data['anchors'])} anchors"
        )
        print(
            f"Encoding {expected_optional} bits using references [{self.key}..{self.key + expected_optional - 1}]"
        )
        print(f"{'=' * 60}")

        base_refs = references[: self.key]
        base_sentences = self._generate_base_sentences(seed_data, base_refs)

        print(f"\nBase sentences ({len(base_sentences)}):")
        for i, s in enumerate(base_sentences):
            print(f"  {i + 1}. {s}")

        print(f"\nGenerating {expected_optional} optional sentences...")

        initial_history = {
            "seed_title": seed_data["title"],
            "seed_abstract": seed_data["abstract"],
            "references": references,
            "key": self.key,
            "base_sentences": base_sentences,
            "optional_sentences": [],
        }

        optional_sentences, _ = self.encode(
            chunks=chunks,
            history=initial_history,
            system_prompt=SENTENCE_CONTINUATION,
            max_length=200,
            temperature=1,
        )

        print(f"\nOptional sentences ({len(optional_sentences)}):")
        for i, s in enumerate(optional_sentences):
            print(f"  {self.key + i + 1}. {s}")

        stego_text = " ".join(base_sentences + optional_sentences)
        # stego_text = self._synthsize(base_sentences + optional_sentences, seed_data)
        return stego_text

    def recover_message(self, stego_text: str, **kwargs):
        if self.message_length is None:
            raise ValueError(
                "No message length set. Run hide_message first or set message_length."
            )

        expected_optional = self.message_length // self.hash_output_length
        expected_total = self.key + expected_optional

        print(f"\n{'=' * 60}")
        print("Recovery - LLM Decomposition")
        print(f"{'=' * 60}")
        print(
            f"  Expected: {expected_total} spans (key={self.key}, optional={expected_optional})"
        )

        spans = self._decompose(stego_text, expected_total)
        print(f"  Extracted: {len(spans)} spans")

        for i, s in enumerate(spans):
            print(f"    {i + 1}. {s[:120]}...")

        if len(spans) != expected_total:
            print(f"  WARNING: Expected {expected_total} spans, got {len(spans)}")

        optional_spans = spans[self.key : self.key + expected_optional]

        if len(optional_spans) < expected_optional:
            print(
                f"  WARNING: Only {len(optional_spans)} optional spans, expected {expected_optional}"
            )

        embeddings = get_embeddings_in_batch(self.client, optional_spans)
        return self._decode_from_embeddings(embeddings, self.message_length)

    def _synthsize(self, sentences: list[str], seed_data: dict) -> str:
        response = generate_response(
            prompt="\n".join([f"{idx}. {s}" for idx, s in enumerate(sentences)]),
            system_prompt=SYNTHESIZE.format(
                seed_title=seed_data["title"], seed_abstract=seed_data["abstract"]
            ),
            max_length=4000,
            temperature=0,
        )

        return response

    def _decompose(self, text: str, expected_total: int) -> list[str]:
        response = generate_response(
            prompt=text,
            system_prompt=DECOMPOSE.format(expected_total=expected_total),
            max_length=4000,
            temperature=0,
        )

        spans = []
        for part in response.split("[sep]"):
            span = part.strip()
            if not span:
                continue
            span = re.sub(r"^\d+[.):\s]+", "", span).strip()
            span = re.sub(r"^[-\u2022*]\s*", "", span).strip()
            if span:
                spans.append(span)

        if len(spans) != expected_total:
            print(
                f"  Decomposition returned {len(spans)} spans, expected {expected_total}"
            )

        return spans

    def _generate_base_sentences(
        self, seed_data: dict, base_refs: list[dict]
    ) -> list[str]:
        refs_formatted = "\n".join(
            f"  {i + 1}. {r['author_text']} ({r['year']}). {r['ref_title']}"
            for i, r in enumerate(base_refs)
        )
        prompt = (
            f'Paper: "{seed_data["title"]}"\n'
            f"Abstract: {seed_data['abstract'][:600]}\n\n"
            f"References to describe (in this order):\n{refs_formatted}"
        )

        response = generate_response(
            prompt=prompt,
            system_prompt=BASE_GENERATION.format(k=len(base_refs)),
            max_length=2000,
            temperature=0,
        )

        sentences = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\d+[.):\s]+", "", line).strip()
            if line:
                sentences.append(line)

        if len(sentences) < len(base_refs):
            print(
                f"  WARNING: Generated {len(sentences)} base sentences, expected {len(base_refs)}"
            )

        return sentences[: len(base_refs)]
