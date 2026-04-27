"""Phase 1: Generate stego (S), same-pipeline cover (C1), and prompted cover (C2)
texts for all three systems.

One S + one C1 + one C2 per prompt, per experiment.md lines 144-153:
    300 prompts × 1 message × 3 text types = 900 texts per system.

Output layout matches experiment.md lines 33-42 (separate files per text_type):

    data/experiments/phase1_texts/
        topicqa_stego.jsonl
        topicqa_cover_c1.jsonl
        topicqa_cover_c2.jsonl
        story_stego.jsonl
        story_cover_c1.jsonl
        story_cover_c2.jsonl
        litreview_stego.jsonl
        litreview_cover_c1.jsonl
        litreview_cover_c2.jsonl

Usage:
    python -m experiments.phase1_generate --system topicqa
    python -m experiments.phase1_generate --system story
    python -m experiments.phase1_generate --system litreview
    python -m experiments.phase1_generate --system all
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.utils.io import (
    append_jsonl,
    load_completed_ids,
    load_records_map,
    make_record_id,
)
from experiments.utils.system_factory import (
    make_clients,
    make_litreview,
    make_story,
    make_topicqa,
)
from experiments.utils.token_counter import count_tokens, count_words, round_words

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _direct_gpt_call(client, prompt: str, max_tokens: int = 4000) -> str:
    """Direct GPT-4.1 call for C2 cover text generation (no steg pipeline)."""
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


def _make_record(
    record_id: str,
    system: str,
    text_type: str,
    prompt_idx: int,
    prompt: str,
    text: str,
    message_bits: list[int] | None,
    system_state: dict | None,
    metadata: dict | None,
    length_target: int | None = None,
    paired_stego_id: str | None = None,
) -> dict:
    return {
        "id": record_id,
        "system": system,
        "text_type": text_type,
        "prompt_idx": prompt_idx,
        "prompt": prompt,
        "message_bits": message_bits,
        "text": text,
        "token_count": count_tokens(text),
        "word_count": count_words(text),
        "char_count": len(text),
        "system_state": system_state,
        "metadata": metadata,
        "length_target": length_target,
        "paired_stego_id": paired_stego_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _out_paths(output_dir: Path, system: str) -> dict[str, Path]:
    return {
        "stego": output_dir / f"{system}_stego.jsonl",
        "cover_c1": output_dir / f"{system}_cover_c1.jsonl",
        "cover_c2": output_dir / f"{system}_cover_c2.jsonl",
    }


def _load_checkpoint(paths: dict[str, Path]) -> tuple[set[str], dict[str, dict]]:
    """Aggregate completed ids and records across all per-type files for resumption."""
    completed: set[str] = set()
    records_map: dict[str, dict] = {}
    for p in paths.values():
        completed |= load_completed_ids(p)
        records_map.update(load_records_map(p))
    return completed, records_map


# ---------------------------------------------------------------------------
# TopicQA generation
# ---------------------------------------------------------------------------


def generate_topicqa(
    client, local_client, prompts: list[dict], messages: dict, output_dir: Path
):
    """Generate TopicQA texts: 1 S + 1 C1 + 1 C2 per prompt (experiment.md Phase 1)."""
    paths = _out_paths(output_dir, "topicqa")
    system = make_topicqa(client, local_client)
    completed, records_map = _load_checkpoint(paths)

    stego_msgs = messages["stego_messages"]
    c1_msgs = messages["c1_messages"]
    n_prompts = len(prompts)

    log.info(f"TopicQA: {n_prompts} prompts, {len(completed)} records already done")

    for p_idx, prompt_data in enumerate(prompts):
        question = prompt_data["question"]
        log.info(f"TopicQA prompt {p_idx + 1}/{n_prompts}: {question[:60]}...")

        # --- Stego text (S) ---
        s_rid = make_record_id("topicqa", "stego", p_idx)
        if s_rid in completed:
            stego_record = records_map[s_rid]
            log.info(f"  Skip {s_rid} (exists)")
        else:
            msg_bits = stego_msgs[p_idx]
            text = system.hide_message(msg_bits, question)
            stego_record = _make_record(
                record_id=s_rid,
                system="topicqa",
                text_type="stego",
                prompt_idx=p_idx,
                prompt=question,
                text=text,
                message_bits=msg_bits,
                system_state={
                    "question": system._question,
                    "error_encoded_length": system._error_encoded_length,
                },
                metadata=system._last_metadata,
            )
            append_jsonl(paths["stego"], stego_record)
            records_map[s_rid] = stego_record
            completed.add(s_rid)
            log.info(f"  Generated {s_rid} ({stego_record['word_count']} words)")

        # --- Same-pipeline cover (C1) ---
        c1_rid = make_record_id("topicqa", "cover_c1", p_idx)
        if c1_rid not in completed:
            c1_bits = c1_msgs[p_idx]
            c1_text = system.hide_message(c1_bits, question)
            c1_record = _make_record(
                record_id=c1_rid,
                system="topicqa",
                text_type="cover_c1",
                prompt_idx=p_idx,
                prompt=question,
                text=c1_text,
                message_bits=c1_bits,
                system_state={
                    "question": system._question,
                    "error_encoded_length": system._error_encoded_length,
                },
                metadata=system._last_metadata,
                paired_stego_id=s_rid,
            )
            append_jsonl(paths["cover_c1"], c1_record)
            completed.add(c1_rid)
            log.info(f"  Generated {c1_rid} ({c1_record['word_count']} words)")
        else:
            log.info(f"  Skip {c1_rid} (exists)")

        # --- Prompted cover (C2); length target from stego ---
        c2_rid = make_record_id("topicqa", "cover_c2", p_idx)
        if c2_rid not in completed:
            target_words = round_words(stego_record["word_count"])
            c2_prompt = (
                f"Answer the following question in approximately {target_words} words."
                f"\n\nQuestion: {question}"
            )
            c2_text = _direct_gpt_call(client, c2_prompt)
            c2_record = _make_record(
                record_id=c2_rid,
                system="topicqa",
                text_type="cover_c2",
                prompt_idx=p_idx,
                prompt=question,
                text=c2_text,
                message_bits=None,
                system_state=None,
                metadata=None,
                length_target=target_words,
                paired_stego_id=s_rid,
            )
            append_jsonl(paths["cover_c2"], c2_record)
            completed.add(c2_rid)
            log.info(
                f"  Generated {c2_rid} ({c2_record['word_count']} words, target={target_words})"
            )
        else:
            log.info(f"  Skip {c2_rid} (exists)")


# ---------------------------------------------------------------------------
# StorySlot generation
# ---------------------------------------------------------------------------


def generate_story(
    client, local_client, prompts: list[dict], messages: dict, output_dir: Path
):
    """Generate StorySlot texts: 1 S + 1 C1 + 1 C2 per prompt."""
    paths = _out_paths(output_dir, "story")
    system = make_story(client, local_client)
    completed, records_map = _load_checkpoint(paths)

    stego_msgs = messages["stego_messages"]
    c1_msgs = messages["c1_messages"]
    n_prompts = len(prompts)

    log.info(f"StorySlot: {n_prompts} prompts, {len(completed)} records already done")

    for p_idx, prompt_data in enumerate(prompts):
        premise = prompt_data["premise"]
        log.info(f"StorySlot prompt {p_idx + 1}/{n_prompts}: {premise[:60]}...")

        # --- Stego text (S) ---
        s_rid = make_record_id("story", "stego", p_idx)
        if s_rid in completed:
            stego_record = records_map[s_rid]
            log.info(f"  Skip {s_rid} (exists)")
        else:
            msg_bits = stego_msgs[p_idx]
            text = system.hide_message(msg_bits, premise)
            stego_record = _make_record(
                record_id=s_rid,
                system="story",
                text_type="stego",
                prompt_idx=p_idx,
                prompt=premise,
                text=text,
                message_bits=msg_bits,
                system_state={
                    "premise": system._premise,
                    "error_encoded_length": system._error_encoded_length,
                },
                metadata=system._last_metadata,
            )
            append_jsonl(paths["stego"], stego_record)
            records_map[s_rid] = stego_record
            completed.add(s_rid)
            log.info(f"  Generated {s_rid} ({stego_record['word_count']} words)")

        # --- Same-pipeline cover (C1) ---
        c1_rid = make_record_id("story", "cover_c1", p_idx)
        if c1_rid not in completed:
            c1_bits = c1_msgs[p_idx]
            c1_text = system.hide_message(c1_bits, premise)
            c1_record = _make_record(
                record_id=c1_rid,
                system="story",
                text_type="cover_c1",
                prompt_idx=p_idx,
                prompt=premise,
                text=c1_text,
                message_bits=c1_bits,
                system_state={
                    "premise": system._premise,
                    "error_encoded_length": system._error_encoded_length,
                },
                metadata=system._last_metadata,
                paired_stego_id=s_rid,
            )
            append_jsonl(paths["cover_c1"], c1_record)
            completed.add(c1_rid)
            log.info(f"  Generated {c1_rid} ({c1_record['word_count']} words)")
        else:
            log.info(f"  Skip {c1_rid} (exists)")

        # --- Prompted cover (C2) ---
        c2_rid = make_record_id("story", "cover_c2", p_idx)
        if c2_rid not in completed:
            target_words = round_words(stego_record["word_count"])
            c2_prompt = (
                f"Write a story of approximately {target_words} words based on the following premise."
                f"\n\nPremise: {premise}"
            )
            c2_text = _direct_gpt_call(client, c2_prompt)
            c2_record = _make_record(
                record_id=c2_rid,
                system="story",
                text_type="cover_c2",
                prompt_idx=p_idx,
                prompt=premise,
                text=c2_text,
                message_bits=None,
                system_state=None,
                metadata=None,
                length_target=target_words,
                paired_stego_id=s_rid,
            )
            append_jsonl(paths["cover_c2"], c2_record)
            completed.add(c2_rid)
            log.info(
                f"  Generated {c2_rid} ({c2_record['word_count']} words, target={target_words})"
            )
        else:
            log.info(f"  Skip {c2_rid} (exists)")


# ---------------------------------------------------------------------------
# LitReview generation
# ---------------------------------------------------------------------------


def generate_litreview(
    client, corpus_indices: list[int], messages: dict, output_dir: Path
):
    """Generate LitReview texts: 1 S + 1 C1 + 1 C2 per prompt."""
    paths = _out_paths(output_dir, "litreview")
    system = make_litreview(client)
    completed, records_map = _load_checkpoint(paths)

    stego_msgs = messages["stego_messages"]
    c1_msgs = messages["c1_messages"]
    n_prompts = len(corpus_indices)

    log.info(f"LitReview: {n_prompts} prompts, {len(completed)} records already done")

    from embeddings.core.litreview_v2 import prepare_references

    failures_path = output_dir / "litreview_failures.jsonl"

    for p_idx, corpus_idx in enumerate(corpus_indices):
        corpus_idx = int(corpus_idx)
        paper = system.corpus[corpus_idx]
        paper_title = paper["title"]
        log.info(
            f"LitReview prompt {p_idx + 1}/{n_prompts}: [{corpus_idx}] {paper_title[:60]}..."
        )

        # --- Stego text (S) ---
        s_rid = make_record_id("litreview", "stego", p_idx)
        if s_rid in completed:
            stego_record = records_map[s_rid]
            log.info(f"  Skip {s_rid} (exists)")
        else:
            msg_bits = stego_msgs[p_idx]
            try:
                text = system.hide_message(msg_bits, corpus_idx)
            except ValueError as e:
                # Greedy ref-selection failure for this (paper, message) pair.
                # Log and skip — we'll top up these slots in a follow-up pass.
                log.warning(f"  SKIP {s_rid} encode failure: {e}")
                append_jsonl(failures_path, {
                    "id": s_rid,
                    "stage": "stego",
                    "prompt_idx": p_idx,
                    "corpus_idx": corpus_idx,
                    "paper_title": paper_title,
                    "message_bits": msg_bits,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue
            stego_record = _make_record(
                record_id=s_rid,
                system="litreview",
                text_type="stego",
                prompt_idx=p_idx,
                prompt=paper_title,
                text=text,
                message_bits=msg_bits,
                system_state={
                    "error_encoded_length": system._error_encoded_length,
                    "corpus_idx": corpus_idx,
                },
                metadata=system._last_metadata,
            )
            append_jsonl(paths["stego"], stego_record)
            records_map[s_rid] = stego_record
            completed.add(s_rid)
            log.info(f"  Generated {s_rid} ({stego_record['word_count']} words)")

        # --- Same-pipeline cover (C1) ---
        c1_rid = make_record_id("litreview", "cover_c1", p_idx)
        if c1_rid not in completed:
            c1_bits = c1_msgs[p_idx]
            try:
                c1_text = system.hide_message(c1_bits, corpus_idx)
            except ValueError as e:
                log.warning(f"  SKIP {c1_rid} encode failure: {e}")
                append_jsonl(failures_path, {
                    "id": c1_rid,
                    "stage": "cover_c1",
                    "prompt_idx": p_idx,
                    "corpus_idx": corpus_idx,
                    "paper_title": paper_title,
                    "message_bits": c1_bits,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                continue
            c1_record = _make_record(
                record_id=c1_rid,
                system="litreview",
                text_type="cover_c1",
                prompt_idx=p_idx,
                prompt=paper_title,
                text=c1_text,
                message_bits=c1_bits,
                system_state={
                    "error_encoded_length": system._error_encoded_length,
                    "corpus_idx": corpus_idx,
                },
                metadata=system._last_metadata,
                paired_stego_id=s_rid,
            )
            append_jsonl(paths["cover_c1"], c1_record)
            completed.add(c1_rid)
            log.info(f"  Generated {c1_rid} ({c1_record['word_count']} words)")
        else:
            log.info(f"  Skip {c1_rid} (exists)")

        # --- Prompted cover (C2) ---
        c2_rid = make_record_id("litreview", "cover_c2", p_idx)
        if c2_rid not in completed:
            target_words = round_words(stego_record["word_count"])
            all_refs = prepare_references(paper["references"])
            ref_list = "\n".join(
                f"- {r['author_text']} ({r['year']}). {r['ref_title']}"
                for r in all_refs
            )
            c2_prompt = f"""Write a realistic Related Work section for an academic paper on the topic of: "{paper_title}
Cite as "LastName (YEAR)" for single authors or "LastName et al. (YEAR)" for multiple authors.
Every provided reference must appear exactly only once.
Write approximately {target_words} words.

References you can use:
{ref_list}"""
            c2_text = _direct_gpt_call(client, c2_prompt, max_tokens=8000)
            c2_record = _make_record(
                record_id=c2_rid,
                system="litreview",
                text_type="cover_c2",
                prompt_idx=p_idx,
                prompt=paper_title,
                text=c2_text,
                message_bits=None,
                system_state=None,
                metadata=None,
                length_target=target_words,
                paired_stego_id=s_rid,
            )
            append_jsonl(paths["cover_c2"], c2_record)
            completed.add(c2_rid)
            log.info(
                f"  Generated {c2_rid} ({c2_record['word_count']} words, target={target_words})"
            )
        else:
            log.info(f"  Skip {c2_rid} (exists)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Text Generation")
    parser.add_argument(
        "--system",
        choices=["topicqa", "story", "litreview", "all"],
        default="all",
        help="Which system(s) to generate texts for",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/experiments"),
        help="Base directory for prompts and output",
    )
    args = parser.parse_args()

    prompts_dir = args.data_dir / "prompts"
    output_dir = args.data_dir / "phase1_texts"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(prompts_dir / "messages.json") as f:
        all_messages = json.load(f)

    client, local_client = make_clients()

    if args.system in ("topicqa", "all"):
        with open(prompts_dir / "topicqa_prompts.json") as f:
            prompts = json.load(f)["prompts"]
        generate_topicqa(
            client, local_client, prompts, all_messages["topicqa"], output_dir
        )

    if args.system in ("story", "all"):
        with open(prompts_dir / "story_prompts.json") as f:
            prompts = json.load(f)["prompts"]
        generate_story(
            client, local_client, prompts, all_messages["story"], output_dir
        )

    if args.system in ("litreview", "all"):
        with open(prompts_dir / "litreview_indices.json") as f:
            indices_data = json.load(f)
        generate_litreview(
            client, indices_data["indices"], all_messages["litreview"], output_dir
        )

    log.info("Phase 1 generation complete.")


if __name__ == "__main__":
    main()
