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
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from embeddings import CORPORATE_MONOLOGUE
from embeddings.core.story_system_v2 import STORY_SYNTHESIS_PROMPT
from embeddings.core.topicQA_system import SUBTOPIC_PROMPT
from experiments.utils.io import (
    append_jsonl,
    load_completed_ids,
    load_records_map,
    make_record_id,
)
from experiments.utils.system_factory import (
    make_baseline,
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


def _direct_local_call(
    local_client,
    model: str,
    prompt: str,
    max_tokens: int = 3000,
    temperature: float = 0.7,
) -> str:
    """Direct local-model call for C2 cover text generation (no steg pipeline).

    Used for the story system's C2 so the cover generator model matches the
    stego pipeline's plot-determining model (Qwen3.5-4B) instead of GPT-4.1.
    Without this, stego (Qwen3.5 plot scaffold) vs C2 (GPT-4.1 plot) confounds
    the slot mechanism with a model-style difference. Thinking is disabled to
    match the slot-generation call configuration.
    """
    for attempt in range(3):
        try:
            r = local_client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                raise
            log.warning(f"Local call retry {attempt + 1}: {e}")
            time.sleep(2**attempt)


# Option B cover: free-form outline analogue of SLOT_GENERATION_PROMPT (no A/B
# structure). Constraints mirror the slot prompt so the only difference between
# stego and this cover is slot-list vs free outline, not the constraint set.
STORY_OUTLINE_PROMPT = """Given the following story premise, generate exactly {n} concrete story beats that together outline the story.
Each beat is one narrative event or detail in the story.

Requirements:
- Each beat must be a clearly distinguishable concrete object, location, method, action, or event (not an abstract quality).
- Beats must focus on plot or setting choices (objects, locations, methods, events, physical descriptions). DO NOT make any beat about a character's name, identity, role, or personal attributes.
- Beats should follow a natural narrative order (setup -> rising action -> climax -> resolution).
- Each beat should be a short phrase (3-10 words).

Output ONLY a JSON array of short strings. No explanation. NO code block or markdown wrapping!!
Example: ["a sealed envelope slipped under the door", "an encrypted flash drive found in a drawer", "a tense confrontation on a rain-soaked rooftop"]

Story premise: {premise}"""


def _parse_json_list(raw: str) -> list[str]:
    """Parse a JSON array of strings into a flat list of beats.

    Tolerates code-fence wrapping and the common local-model failure mode of
    emitting one single-element array (or a bare quoted string) per line
    instead of a single array spanning the whole output.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    def _flatten(obj) -> list[str]:
        if isinstance(obj, str):
            return [obj.strip()] if obj.strip() else []
        if isinstance(obj, list):
            out: list[str] = []
            for x in obj:
                out.extend(_flatten(x))
            return out
        return []

    # 1. Whole output as one JSON array spanning the full text.
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        try:
            return _flatten(json.loads(raw[start : end + 1]))
        except json.JSONDecodeError:
            pass

    # 2. Per-line fallback: one JSON value (array or bare string) per line.
    items: list[str] = []
    for line in raw.splitlines():
        line = line.strip().rstrip(",").strip()
        if not line:
            continue
        try:
            items.extend(_flatten(json.loads(line)))
        except json.JSONDecodeError:
            continue
    if items:
        return items

    # 3. Last resort: pull quoted strings directly out of the raw text.
    return [m.strip() for m in re.findall(r'"([^"]{3,})"', raw) if m.strip()]


def _generate_story_outline_cover(
    client, local_client, local_model: str, premise: str, n_beats: int
) -> tuple[str, list[str]]:
    """Option B cover (C3): Qwen generates a free-form outline, GPT-4.1 synthesizes.

    Holds the plot-origin model (Qwen) and the prose model (GPT-4.1) constant
    with the stego pipeline. The only difference vs stego is that the plot
    enters synthesis as a free outline rather than an A/B slot list — this
    isolates whether the slot-encoding mechanism itself leaves a trace.
    Uses the same STORY_SYNTHESIS_PROMPT as the stego pipeline.
    """
    raw = _direct_local_call(
        local_client,
        local_model,
        STORY_OUTLINE_PROMPT.format(n=n_beats, premise=premise),
        max_tokens=2000,
        temperature=0,
    )
    print(raw)
    beats = _parse_json_list(raw)
    if len(beats) < n_beats:
        log.warning(f"  outline returned {len(beats)} beats, expected {n_beats}")
    events_str = "\n".join(f"- {b}" for b in beats)
    synth_prompt = STORY_SYNTHESIS_PROMPT.format(premise=premise, events_str=events_str)
    text = _direct_gpt_call(client, synth_prompt)
    return text, beats


# Option B / C3 response prompt for TopicQA: structurally mirrors ENCODE_PROMPT
# but drops the forbidden block, since C3 has no bit-driven include/forbid
# selection. The aspect list comes from Qwen directly (n_groups items, matching
# the "selected" count fed to GPT-4.1 in stego).
TOPICQA_FREE_RESPONSE_PROMPT = """Answer the following question as natural, cohesive prose. Do not use bullet points, numbered lists, section headers, or bold text.

Your answer must substantively cover each of these aspects:
{topics_str}

Question: {question}"""


def _generate_topicqa_free_cover(
    client, local_client, local_model: str, question: str, n_aspects: int
) -> tuple[str, list[str]]:
    """Option B cover (C3): Qwen generates aspects, GPT-4.1 writes the answer.

    Holds the aspect-generator model (Qwen) and the response model (GPT-4.1)
    constant with the stego pipeline. The only difference vs stego is that
    aspects enter response generation as a free list rather than via the
    grouped, bit-driven include/forbid mechanism — this isolates whether the
    bit-encoding mechanism itself leaves a trace.
    """
    raw = _direct_local_call(
        local_client,
        local_model,
        SUBTOPIC_PROMPT.format(n=n_aspects, question=question),
        max_tokens=2500,
        temperature=0,
    )
    aspects = _parse_json_list(raw)
    if len(aspects) < n_aspects:
        log.warning(
            f"  topicqa aspect generation returned {len(aspects)} aspects, expected {n_aspects}"
        )
    aspects = aspects[:n_aspects]
    topics_str = "\n".join(f"- {a}" for a in aspects)
    response_prompt = TOPICQA_FREE_RESPONSE_PROMPT.format(
        topics_str=topics_str, question=question
    )
    text = _direct_gpt_call(client, response_prompt)
    return text, aspects


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
        "cover_c3": output_dir / f"{system}_cover_c3.jsonl",
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
    client,
    local_client,
    prompts: list[dict],
    messages: dict,
    output_dir: Path,
    stego_only: bool = False,
    n_subtopics: int = 12,
    group_size: int = 2,
):
    """Generate TopicQA texts: 1 S + 1 C1 + 1 C2 per prompt (experiment.md Phase 1)."""
    paths = _out_paths(output_dir, "topicqa")
    system = make_topicqa(client, local_client, n_subtopics=n_subtopics, group_size=group_size)
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

        if stego_only:
            continue

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
                f"Answer the following question in approximately {target_words} words.\n\n"
                f"Write the response as cohesive flowing prose. "
                f"Do not use bullet points, numbered lists, section headers, or bold text.\n\n"
                f"Question: {question}"
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

        # --- Option B cover (C3): Qwen aspects + GPT-4.1 response, no bit selection ---
        c3_rid = make_record_id("topicqa", "cover_c3", p_idx)
        if c3_rid not in completed:
            c3_text, c3_aspects = _generate_topicqa_free_cover(
                client, local_client, system.local_model, question, system.n_groups
            )
            c3_record = _make_record(
                record_id=c3_rid,
                system="topicqa",
                text_type="cover_c3",
                prompt_idx=p_idx,
                prompt=question,
                text=c3_text,
                message_bits=None,
                system_state=None,
                metadata={
                    "aspects": c3_aspects,
                    "aspect_generator": system.local_model,
                },
                length_target=round_words(stego_record["word_count"]),
                paired_stego_id=s_rid,
            )
            append_jsonl(paths["cover_c3"], c3_record)
            completed.add(c3_rid)
            log.info(f"  Generated {c3_rid} ({c3_record['word_count']} words)")
        else:
            log.info(f"  Skip {c3_rid} (exists)")


# ---------------------------------------------------------------------------
# StorySlot generation
# ---------------------------------------------------------------------------


def generate_story(
    client,
    local_client,
    prompts: list[dict],
    messages: dict,
    output_dir: Path,
    stego_only: bool = False,
    n_slots: int = 20,
):
    """Generate StorySlot texts: 1 S + 1 C1 + 1 C2 per prompt."""
    paths = _out_paths(output_dir, "story")
    system = make_story(client, local_client, n_slots=n_slots)
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

        if stego_only:
            continue

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
                f"Write a short story of approximately {target_words} words based on the following premise.\n"
                f"Use flowing prose with natural pacing. Develop the story through concrete actions, "
                f"physical detail, and brief dialogue only where it advances the plot. Vary paragraph length naturally.\n"
                f"Build the story around 15-20 specific concrete elements (locations, objects, methods, "
                f"character actions). Each scene should be grounded in physical detail rather than introspection.\n\n"
                f"Premise: {premise}"
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

        # --- Option B cover (C3): Qwen outline + GPT-4.1 synthesis ---
        c3_rid = make_record_id("story", "cover_c3", p_idx)
        if c3_rid not in completed:
            c3_text, c3_beats = _generate_story_outline_cover(
                client, local_client, system.local_model, premise, system.n_slots
            )
            c3_record = _make_record(
                record_id=c3_rid,
                system="story",
                text_type="cover_c3",
                prompt_idx=p_idx,
                prompt=premise,
                text=c3_text,
                message_bits=None,
                system_state=None,
                metadata={
                    "outline": c3_beats,
                    "outline_generator": system.local_model,
                },
                length_target=round_words(stego_record["word_count"]),
                paired_stego_id=s_rid,
            )
            append_jsonl(paths["cover_c3"], c3_record)
            completed.add(c3_rid)
            log.info(f"  Generated {c3_rid} ({c3_record['word_count']} words)")
        else:
            log.info(f"  Skip {c3_rid} (exists)")


# ---------------------------------------------------------------------------
# LitReview generation
# ---------------------------------------------------------------------------


def generate_litreview(
    client,
    corpus_indices: list[int],
    messages: dict,
    output_dir: Path,
    stego_only: bool = False,
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
                append_jsonl(
                    failures_path,
                    {
                        "id": s_rid,
                        "stage": "stego",
                        "prompt_idx": p_idx,
                        "corpus_idx": corpus_idx,
                        "paper_title": paper_title,
                        "message_bits": msg_bits,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
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

        if stego_only:
            continue

        # --- Same-pipeline cover (C1) ---
        c1_rid = make_record_id("litreview", "cover_c1", p_idx)
        if c1_rid not in completed:
            c1_bits = c1_msgs[p_idx]
            try:
                c1_text = system.hide_message(c1_bits, corpus_idx)
            except ValueError as e:
                log.warning(f"  SKIP {c1_rid} encode failure: {e}")
                append_jsonl(
                    failures_path,
                    {
                        "id": c1_rid,
                        "stage": "cover_c1",
                        "prompt_idx": p_idx,
                        "corpus_idx": corpus_idx,
                        "paper_title": paper_title,
                        "message_bits": c1_bits,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
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
            n_refs_target = len(stego_record["metadata"]["selected_refs"])
            all_refs = prepare_references(paper["references"])
            ref_list = "\n".join(
                f"- {r['author_text']} ({r['year']}). {r['ref_title']}"
                for r in all_refs
            )
            seed_abstract = (paper.get("abstract") or "")[:600]
            c2_prompt = (
                f"You are writing the Related Work section of an academic paper.\n"
                f"Paper: {paper_title}\n"
                f"Abstract: {seed_abstract}\n"
                f"Write a Related Work section that contextualizes this paper within the broader research landscape. Organize thematically, grouping related works by research direction or methodology across multiple paragraphs."
                f"Where works are closely related, discuss them together in the same sentence or passage rather than giving each its own isolated sentence. Include contextual sentences that provide background or transitions without citing specific papers. Some works may warrant more discussion than others depending on their relevance.\n"
                f"""Cite as "LastName (YEAR)" or "LastName et al. (YEAR)". Each cited reference should appear exactly once."""
                f"Select approximately {n_refs_target} references from the list below that best fit the section — choose references that flow naturally together; you do NOT need to use all of them.\n"
                f"Length: approximately {target_words} words.\n"
                f"Available references:\n"
                f"{ref_list}"
            )
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
# Baseline generation
# ---------------------------------------------------------------------------


def generate_baseline(
    client,
    prompts: list[dict],
    messages: dict,
    output_dir: Path,
    stego_only: bool = False,
):
    """Generate Baseline (SentenceStegSystem) texts: 1 S + 1 C1 + 1 C2 per prompt."""
    paths = _out_paths(output_dir, "baseline")
    system = make_baseline(client)
    completed, records_map = _load_checkpoint(paths)

    stego_msgs = messages["stego_messages"]
    c1_msgs = messages["c1_messages"]
    n_prompts = len(prompts)

    log.info(f"Baseline: {n_prompts} prompts, {len(completed)} records already done")

    for p_idx, prompt_data in enumerate(prompts):
        seed = prompt_data["seed"]
        log.info(f"Baseline prompt {p_idx + 1}/{n_prompts}: {seed[:60]}...")

        # --- Stego text (S) ---
        s_rid = make_record_id("baseline", "stego", p_idx)
        if s_rid in completed:
            stego_record = records_map[s_rid]
            log.info(f"  Skip {s_rid} (exists)")
        else:
            msg_bits = stego_msgs[p_idx]
            text = system.hide_message(msg_bits, seed)
            stego_record = _make_record(
                record_id=s_rid,
                system="baseline",
                text_type="stego",
                prompt_idx=p_idx,
                prompt=seed,
                text=text,
                message_bits=msg_bits,
                system_state={
                    "seed": system._seed,
                    "error_encoded_length": system._error_encoded_length,
                },
                metadata=system._last_metadata,
            )
            append_jsonl(paths["stego"], stego_record)
            records_map[s_rid] = stego_record
            completed.add(s_rid)
            log.info(f"  Generated {s_rid} ({stego_record['word_count']} words)")

        if stego_only:
            continue

        # --- Same-pipeline cover (C1) ---
        c1_rid = make_record_id("baseline", "cover_c1", p_idx)
        if c1_rid not in completed:
            c1_bits = c1_msgs[p_idx]
            c1_text = system.hide_message(c1_bits, seed)
            c1_record = _make_record(
                record_id=c1_rid,
                system="baseline",
                text_type="cover_c1",
                prompt_idx=p_idx,
                prompt=seed,
                text=c1_text,
                message_bits=c1_bits,
                system_state={
                    "seed": system._seed,
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
        c2_rid = make_record_id("baseline", "cover_c2", p_idx)
        if c2_rid not in completed:
            target_words = round_words(stego_record["word_count"])
            c2_prompt = (
                f"Write a corporate email-style passage of approximately {target_words} words.\n\n"
                f"Write the response as cohesive flowing prose. "
                f"Do not use bullet points, numbered lists, section headers, or bold text.\n\n"
                f"{CORPORATE_MONOLOGUE}\n\n"
                f"Context / opening: {seed}"
            )
            c2_text = _direct_gpt_call(client, c2_prompt)
            c2_record = _make_record(
                record_id=c2_rid,
                system="baseline",
                text_type="cover_c2",
                prompt_idx=p_idx,
                prompt=seed,
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
        choices=["topicqa", "story", "litreview", "baseline", "all"],
        default="all",
        help="Which system(s) to generate texts for",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/experiments"),
        help="Base directory for prompts and output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only generate texts for the first N prompts per system (default: all).",
    )
    parser.add_argument(
        "--stego-only",
        action="store_true",
        help="Generate only stego (S) texts; skip all cover texts (C1/C2/C3).",
    )
    parser.add_argument(
        "--subdir",
        default="recovery_test",
        help=(
            "Sub-directory under phase1_texts/ to write outputs to "
            "(default: recovery_test => data/experiments/phase1_texts/recovery_test/). "
            "Pass --subdir '' to write to phase1_texts/ directly. "
            "If --capacity is set and --subdir is left at the default, "
            "subdir auto-becomes '{system}_cap{N}'."
        ),
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help=(
            "Override message-bit count for the chosen system. "
            "Requires --system != all. Auto-sets --subdir to '{system}_cap{N}' "
            "unless --subdir is given explicitly. For topicqa, also sets "
            "n_subtopics = capacity * group_size (unless --n-subtopics is given). "
            "Messages are regenerated inline via np.default_rng(42 + capacity) and "
            "written to {output_dir}/messages.json for traceability."
        ),
    )
    parser.add_argument(
        "--n-subtopics",
        type=int,
        default=None,
        help=(
            "TopicQA only: number of subtopics (capacity = n_subtopics // group_size). "
            "Defaults to 12 (or 2*capacity when --capacity is set)."
        ),
    )
    parser.add_argument(
        "--n-slots",
        type=int,
        default=None,
        help="Story only: number of plot slots (capacity = n_slots). Defaults to 20.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=2,
        help="TopicQA only: group size (power of 2). Bits per group = log2(group_size).",
    )
    args = parser.parse_args()

    # --- Capacity handling: requires a specific system; auto-subdir; inline messages ---
    if args.capacity is not None:
        if args.system == "all":
            parser.error("--capacity requires --system to be one of topicqa/story/litreview/baseline (not 'all').")
        if args.subdir == "recovery_test":
            args.subdir = f"{args.system}_cap{args.capacity}"
            log.info(f"--capacity set: defaulting --subdir to {args.subdir!r}")

    prompts_dir = args.data_dir / "prompts"
    output_dir = args.data_dir / "phase1_texts"
    if args.subdir:
        output_dir = output_dir / args.subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir}")
    if args.limit is not None:
        log.info(f"Limiting to first {args.limit} prompt(s) per system")
    if args.stego_only:
        log.info("Stego-only mode: cover texts (C1/C2/C3) skipped")

    # --- Messages: inline regen for capacity variant, else load shared messages.json ---
    if args.capacity is not None:
        # Validate per-system capacity constraints early.
        if args.system == "topicqa":
            n_subtopics_use = args.n_subtopics if args.n_subtopics is not None else args.capacity * args.group_size
            expected_bits = (n_subtopics_use // args.group_size) * int(np.log2(args.group_size))
            if expected_bits != args.capacity:
                parser.error(
                    f"topicqa capacity mismatch: n_subtopics={n_subtopics_use}, group_size={args.group_size} "
                    f"=> {expected_bits} bits, but --capacity={args.capacity}."
                )

        # Deterministic seed: same prompt set used across all variants, but
        # bits differ per capacity (different num_bits means different draws).
        rng = np.random.default_rng(42 + args.capacity)
        n_prompts_msg = 300  # matches expand_prompts.TARGET_MESSAGES
        stego = rng.integers(0, 2, size=(n_prompts_msg, args.capacity)).tolist()
        c1 = rng.integers(0, 2, size=(n_prompts_msg, args.capacity)).tolist()
        all_messages = {
            args.system: {
                "num_bits": args.capacity,
                "stego_messages": stego,
                "c1_messages": c1,
            }
        }
        # Persist for traceability + Phase 4 sanity checks.
        msgs_path = output_dir / "messages.json"
        msgs_path.write_text(json.dumps({
            "seed": 42 + args.capacity,
            "rng": "numpy.default_rng",
            "system": args.system,
            "capacity": args.capacity,
            **all_messages,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        log.info(f"Wrote variant messages.json ({args.capacity} bits, {n_prompts_msg} per type) to {msgs_path}")
    else:
        with open(prompts_dir / "messages.json") as f:
            all_messages = json.load(f)

    client, local_client = make_clients()

    if args.system in ("topicqa", "all"):
        with open(prompts_dir / "topicqa_prompts.json") as f:
            prompts = json.load(f)["prompts"]
        if args.limit is not None:
            prompts = prompts[: args.limit]
        topicqa_n_subtopics = args.n_subtopics
        if topicqa_n_subtopics is None and args.capacity is not None and args.system == "topicqa":
            topicqa_n_subtopics = args.capacity * args.group_size
        if topicqa_n_subtopics is None:
            topicqa_n_subtopics = 12
        generate_topicqa(
            client,
            local_client,
            prompts,
            all_messages["topicqa"],
            output_dir,
            stego_only=args.stego_only,
            n_subtopics=topicqa_n_subtopics,
            group_size=args.group_size,
        )

    if args.system in ("story", "all"):
        with open(prompts_dir / "story_prompts.json") as f:
            prompts = json.load(f)["prompts"]
        if args.limit is not None:
            prompts = prompts[: args.limit]
        story_n_slots = args.n_slots if args.n_slots is not None else 20
        generate_story(
            client,
            local_client,
            prompts,
            all_messages["story"],
            output_dir,
            stego_only=args.stego_only,
            n_slots=story_n_slots,
        )

    if args.system in ("litreview", "all"):
        with open(prompts_dir / "litreview_indices.json") as f:
            indices_data = json.load(f)
        indices = indices_data["indices"]
        if args.limit is not None:
            indices = indices[: args.limit]
        generate_litreview(
            client,
            indices,
            all_messages["litreview"],
            output_dir,
            stego_only=args.stego_only,
        )

    if args.system in ("baseline", "all"):
        with open(prompts_dir / "baseline_prompts.json") as f:
            prompts = json.load(f)["prompts"]
        if args.limit is not None:
            prompts = prompts[: args.limit]
        generate_baseline(
            client,
            prompts,
            all_messages["baseline"],
            output_dir,
            stego_only=args.stego_only,
        )

    log.info("Phase 1 generation complete.")


if __name__ == "__main__":
    main()
