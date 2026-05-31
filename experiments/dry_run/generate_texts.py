"""Dry-run text generation: 10 stego + 10 C1 + 10 C2 per system.

Fully standalone — no imports from experiments/utils/.

Usage:
    python -m experiments.dry_run.generate_texts --system topicqa
    python -m experiments.dry_run.generate_texts --system story
    python -m experiments.dry_run.generate_texts --system litreview
    python -m experiments.dry_run.generate_texts --system all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import openai

from systems import (
    BypassEncoder,
    LitReviewSystem,
    RepetitionCode,
    StorySystem,
    TopicQASystem,
)
from systems.core.litreview import load_corpus, prepare_references
from systems.paths import litreview_references

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
LOCAL_MODEL = "Qwen3.5-4B-UD-Q8_K_XL.gguf"

DRY_RUN_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = DRY_RUN_DIR / "prompts"
RESULTS_DIR = DRY_RUN_DIR / "results"

BIT_LENGTHS = {"topicqa": 6, "story": 18, "litreview": 15}


# ---------------------------------------------------------------------------
# Self-contained helpers
# ---------------------------------------------------------------------------


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_completed_ids(path: Path) -> set[str]:
    return {r["record_id"] for r in read_jsonl(path) if "record_id" in r}


def load_records_map(path: Path) -> dict[str, dict]:
    return {r["record_id"]: r for r in read_jsonl(path) if "record_id" in r}


def make_record_id(system: str, text_type: str, prompt_idx: int) -> str:
    return f"{system}_{text_type}_p{prompt_idx:03d}"


def count_words(text: str) -> int:
    return len(text.split())


def round_words(n: int, step: int = 50) -> int:
    return max(step, round(n / step) * step)


def generate_bits(rng: random.Random, n_bits: int, n_prompts: int) -> tuple[list, list]:
    stego = [[rng.randint(0, 1) for _ in range(n_bits)] for _ in range(n_prompts)]
    c1 = [[rng.randint(0, 1) for _ in range(n_bits)] for _ in range(n_prompts)]
    return stego, c1


def direct_gpt_call(client: openai.OpenAI, prompt: str, max_tokens: int = 4000) -> str:
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model="gpt-4.1",
                temperature=0.7,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                raise
            log.warning(f"GPT call retry {attempt + 1}: {e}")
            time.sleep(2**attempt)


def make_record(
    record_id: str,
    system_name: str,
    text_type: str,
    prompt_idx: int,
    prompt: str,
    message_bits: list[int] | None,
    text: str,
    system_state: dict | None,
    metadata: dict | None,
    length_target: int | None = None,
) -> dict:
    return {
        "record_id": record_id,
        "system_name": system_name,
        "text_type": text_type,
        "prompt_idx": prompt_idx,
        "prompt": prompt,
        "message_bits": message_bits,
        "text": text,
        "word_count": count_words(text),
        "system_state": system_state,
        "metadata": metadata,
        "length_target": length_target,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# System factories
# ---------------------------------------------------------------------------


def _make_clients() -> tuple[openai.OpenAI, openai.OpenAI]:
    client = openai.OpenAI()
    local_client = openai.OpenAI(base_url=LOCAL_BASE_URL, api_key="unused")
    return client, local_client


def _make_topicqa(client, local_client):
    return TopicQASystem(
        client,
        error_correction=RepetitionCode(1),
        local_client=local_client,
        local_model=LOCAL_MODEL,
        n_subtopics=12,
        group_size=2,
        response_model="gpt-4.1",
        decoder_model="gpt-4.1",
        key="default",
        encoder=BypassEncoder(),
        response_temperature=0.7,
    )


def _make_story(client, local_client):
    return StorySystem(
        client,
        error_correction=RepetitionCode(1),
        local_client=local_client,
        local_model=LOCAL_MODEL,
        n_slots=20,
        response_model="gpt-4.1",
        decoder_model="gpt-4.1",
        key="default",
        encoder=BypassEncoder(),
        response_temperature=0.7,
    )


def _make_litreview(client):
    corpus = load_corpus(*litreview_references())
    return LitReviewSystem(
        client,
        error_correction=RepetitionCode(1),
        corpus=corpus,
        model="gpt-4.1",
        encoder=BypassEncoder(),
    )


# ---------------------------------------------------------------------------
# TopicQA generation
# ---------------------------------------------------------------------------


def generate_topicqa(client, local_client, output_path: Path):
    system = _make_topicqa(client, local_client)
    completed = load_completed_ids(output_path)
    records_map = load_records_map(output_path)

    with open(PROMPTS_DIR / "topicqa_questions.json") as f:
        questions = json.load(f)["questions"]

    rng = random.Random(42)
    stego_msgs, c1_msgs = generate_bits(rng, BIT_LENGTHS["topicqa"], len(questions))

    log.info(f"TopicQA: {len(questions)} prompts, {len(completed)} already completed")

    for p_idx, q_data in enumerate(questions):
        question = q_data["question"]
        log.info(f"TopicQA [{p_idx + 1}/{len(questions)}]: {question[:60]}...")

        # --- Stego ---
        s_rid = make_record_id("topicqa", "stego", p_idx)
        if s_rid in completed:
            stego_rec = records_map[s_rid]
            log.info(f"  Skip {s_rid} (exists)")
        else:
            text = system.hide_message(stego_msgs[p_idx], question)
            stego_rec = make_record(
                record_id=s_rid,
                system_name="topicqa",
                text_type="stego",
                prompt_idx=p_idx,
                prompt=question,
                message_bits=stego_msgs[p_idx],
                text=text,
                system_state={
                    "_question": system._question,
                    "_error_encoded_length": system._error_encoded_length,
                },
                metadata=system._last_metadata,
            )
            append_jsonl(output_path, stego_rec)
            records_map[s_rid] = stego_rec
            completed.add(s_rid)
            log.info(f"  Generated {s_rid} ({stego_rec['word_count']} words)")

        # --- C1 ---
        c1_rid = make_record_id("topicqa", "cover_c1", p_idx)
        if c1_rid not in completed:
            text = system.hide_message(c1_msgs[p_idx], question)
            c1_rec = make_record(
                record_id=c1_rid,
                system_name="topicqa",
                text_type="cover_c1",
                prompt_idx=p_idx,
                prompt=question,
                message_bits=c1_msgs[p_idx],
                text=text,
                system_state={
                    "_question": system._question,
                    "_error_encoded_length": system._error_encoded_length,
                },
                metadata=system._last_metadata,
            )
            append_jsonl(output_path, c1_rec)
            completed.add(c1_rid)
            log.info(f"  Generated {c1_rid} ({c1_rec['word_count']} words)")
        else:
            log.info(f"  Skip {c1_rid} (exists)")

        # --- C2 ---
        c2_rid = make_record_id("topicqa", "cover_c2", p_idx)
        if c2_rid not in completed:
            target_words = round_words(stego_rec["word_count"])
            c2_prompt = (
                f"Answer the following question in approximately {target_words} words."
                f"\n\nQuestion: {question}"
            )
            text = direct_gpt_call(client, c2_prompt)
            c2_rec = make_record(
                record_id=c2_rid,
                system_name="topicqa",
                text_type="cover_c2",
                prompt_idx=p_idx,
                prompt=question,
                message_bits=None,
                text=text,
                system_state=None,
                metadata=None,
                length_target=target_words,
            )
            append_jsonl(output_path, c2_rec)
            completed.add(c2_rid)
            log.info(
                f"  Generated {c2_rid} ({c2_rec['word_count']} words, target={target_words})"
            )
        else:
            log.info(f"  Skip {c2_rid} (exists)")


# ---------------------------------------------------------------------------
# StorySlot generation
# ---------------------------------------------------------------------------


def generate_story(client, local_client, output_path: Path):
    system = _make_story(client, local_client)
    completed = load_completed_ids(output_path)
    records_map = load_records_map(output_path)

    with open(PROMPTS_DIR / "story_premises.json") as f:
        premises = json.load(f)["premises"]

    rng = random.Random(42)
    stego_msgs, c1_msgs = generate_bits(rng, BIT_LENGTHS["story"], len(premises))

    log.info(f"StorySlot: {len(premises)} prompts, {len(completed)} already completed")

    for p_idx, p_data in enumerate(premises):
        premise = p_data["premise"]
        log.info(f"StorySlot [{p_idx + 1}/{len(premises)}]: {premise[:60]}...")

        # --- Stego ---
        s_rid = make_record_id("story", "stego", p_idx)
        if s_rid in completed:
            stego_rec = records_map[s_rid]
            log.info(f"  Skip {s_rid} (exists)")
        else:
            text = system.hide_message(stego_msgs[p_idx], premise)
            stego_rec = make_record(
                record_id=s_rid,
                system_name="story",
                text_type="stego",
                prompt_idx=p_idx,
                prompt=premise,
                message_bits=stego_msgs[p_idx],
                text=text,
                system_state={
                    "_premise": system._premise,
                    "_error_encoded_length": system._error_encoded_length,
                },
                metadata=system._last_metadata,
            )
            append_jsonl(output_path, stego_rec)
            records_map[s_rid] = stego_rec
            completed.add(s_rid)
            log.info(f"  Generated {s_rid} ({stego_rec['word_count']} words)")

        # --- C1 ---
        c1_rid = make_record_id("story", "cover_c1", p_idx)
        if c1_rid not in completed:
            text = system.hide_message(c1_msgs[p_idx], premise)
            c1_rec = make_record(
                record_id=c1_rid,
                system_name="story",
                text_type="cover_c1",
                prompt_idx=p_idx,
                prompt=premise,
                message_bits=c1_msgs[p_idx],
                text=text,
                system_state={
                    "_premise": system._premise,
                    "_error_encoded_length": system._error_encoded_length,
                },
                metadata=system._last_metadata,
            )
            append_jsonl(output_path, c1_rec)
            completed.add(c1_rid)
            log.info(f"  Generated {c1_rid} ({c1_rec['word_count']} words)")
        else:
            log.info(f"  Skip {c1_rid} (exists)")

        # --- C2 ---
        c2_rid = make_record_id("story", "cover_c2", p_idx)
        if c2_rid not in completed:
            target_words = round_words(stego_rec["word_count"])
            c2_prompt = (
                f"Write a story of approximately {target_words} words based on the following premise."
                f"\n\nPremise: {premise}"
            )
            text = direct_gpt_call(client, c2_prompt)
            c2_rec = make_record(
                record_id=c2_rid,
                system_name="story",
                text_type="cover_c2",
                prompt_idx=p_idx,
                prompt=premise,
                message_bits=None,
                text=text,
                system_state=None,
                metadata=None,
                length_target=target_words,
            )
            append_jsonl(output_path, c2_rec)
            completed.add(c2_rid)
            log.info(
                f"  Generated {c2_rid} ({c2_rec['word_count']} words, target={target_words})"
            )
        else:
            log.info(f"  Skip {c2_rid} (exists)")


# ---------------------------------------------------------------------------
# LitReview generation
# ---------------------------------------------------------------------------


def generate_litreview(client, output_path: Path):
    system = _make_litreview(client)
    completed = load_completed_ids(output_path)
    records_map = load_records_map(output_path)

    with open(PROMPTS_DIR / "litreview_indices.json") as f:
        indices = json.load(f)["indices"]

    rng = random.Random(42)
    stego_msgs, c1_msgs = generate_bits(rng, BIT_LENGTHS["litreview"], len(indices))

    log.info(f"LitReview: {len(indices)} prompts, {len(completed)} already completed")

    for p_idx, corpus_idx in enumerate(indices):
        paper = system.corpus[corpus_idx]
        paper_title = paper["title"]
        log.info(
            f"LitReview [{p_idx + 1}/{len(indices)}]: [{corpus_idx}] {paper_title[:60]}..."
        )

        # --- Stego ---
        s_rid = make_record_id("litreview", "stego", p_idx)
        if s_rid in completed:
            stego_rec = records_map[s_rid]
            log.info(f"  Skip {s_rid} (exists)")
        else:
            text = system.hide_message(stego_msgs[p_idx], str(corpus_idx))
            stego_rec = make_record(
                record_id=s_rid,
                system_name="litreview",
                text_type="stego",
                prompt_idx=p_idx,
                prompt=paper_title,
                message_bits=stego_msgs[p_idx],
                text=text,
                system_state={
                    "_error_encoded_length": system._error_encoded_length,
                    "_corpus_idx": corpus_idx,
                },
                metadata=system._last_metadata,
            )
            append_jsonl(output_path, stego_rec)
            records_map[s_rid] = stego_rec
            completed.add(s_rid)
            log.info(f"  Generated {s_rid} ({stego_rec['word_count']} words)")

        # --- C1 ---
        c1_rid = make_record_id("litreview", "cover_c1", p_idx)
        if c1_rid not in completed:
            text = system.hide_message(c1_msgs[p_idx], str(corpus_idx))
            c1_rec = make_record(
                record_id=c1_rid,
                system_name="litreview",
                text_type="cover_c1",
                prompt_idx=p_idx,
                prompt=paper_title,
                message_bits=c1_msgs[p_idx],
                text=text,
                system_state={
                    "_error_encoded_length": system._error_encoded_length,
                    "_corpus_idx": corpus_idx,
                },
                metadata=system._last_metadata,
            )
            append_jsonl(output_path, c1_rec)
            completed.add(c1_rid)
            log.info(f"  Generated {c1_rid} ({c1_rec['word_count']} words)")
        else:
            log.info(f"  Skip {c1_rid} (exists)")

        # --- C2 ---
        c2_rid = make_record_id("litreview", "cover_c2", p_idx)
        if c2_rid not in completed:
            target_words = round_words(stego_rec["word_count"])
            all_refs = prepare_references(paper["references"])
            c2_refs = rng.sample(all_refs, min(15, len(all_refs)))
            ref_list = "\n".join(
                f"- {r['author_text']} ({r['year']}). {r['ref_title']}" for r in c2_refs
            )
            c2_prompt = f"""Write a realistic Related Work section for an academic paper on the topic of: "{paper_title}"
Cite as "LastName (YEAR)" for single authors or "LastName et al. (YEAR)" for multiple authors.
Every provided reference must appear exactly only once.
Write approximately {target_words} words.

References you can use:
{ref_list}"""
            text = direct_gpt_call(client, c2_prompt)
            c2_rec = make_record(
                record_id=c2_rid,
                system_name="litreview",
                text_type="cover_c2",
                prompt_idx=p_idx,
                prompt=paper_title,
                message_bits=None,
                text=text,
                system_state=None,
                metadata=None,
                length_target=target_words,
            )
            append_jsonl(output_path, c2_rec)
            completed.add(c2_rid)
            log.info(
                f"  Generated {c2_rid} ({c2_rec['word_count']} words, target={target_words})"
            )
        else:
            log.info(f"  Skip {c2_rid} (exists)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Dry-run text generation")
    parser.add_argument(
        "--system",
        choices=["topicqa", "story", "litreview", "all"],
        default="all",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client, local_client = _make_clients()

    if args.system in ("topicqa", "all"):
        generate_topicqa(client, local_client, RESULTS_DIR / "topicqa_texts.jsonl")

    if args.system in ("story", "all"):
        generate_story(client, local_client, RESULTS_DIR / "story_texts.jsonl")

    if args.system in ("litreview", "all"):
        generate_litreview(client, RESULTS_DIR / "litreview_texts.jsonl")

    log.info("Dry-run generation complete.")


if __name__ == "__main__":
    main()
