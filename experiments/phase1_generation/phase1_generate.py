"""Phase 1: Generate stegotexts.

One stegotext per prompt, written to
``data/experiments/phase1_texts/{subdir}/{system}_stego.jsonl``. The normal
generations steganalysis compares them with come from
``phase1_normal`` (length matched to these stegotexts), so they are not made
here. The Type-1 covertext (same pipeline, random message) is retired: with a
uniform message it has the stegotext's distribution by construction.

Usage:
    python -m experiments.phase1_generation.phase1_generate --system story
    python -m experiments.phase1_generation.phase1_generate --system litreview

    # Native-capacity variant (auto-subdir {system}_cap{N}):
    python -m experiments.phase1_generation.phase1_generate --system story --capacity 8

    # A payload-grid cell (subdir recovery/story_cap16_syn-..._gen-...):
    python -m experiments.phase1_generation.phase1_generate --system story \
        --capacity 16 --track recovery --synth-provider together \
        --synth-model deepseek-ai/DeepSeek-V4.1-Flash --generator-model ...
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.utils.configs import CONFIG_SYSTEMS, config_tag, default_config
from experiments.utils.io import (
    TRACKS,
    append_jsonl,
    load_records_map,
    make_record_id,
)
from experiments.utils.system_factory import (
    PROVIDERS,
    make_client,
    make_clients,
    make_discop,
    make_litreview,
    make_story,
    default_sampling_only,
    writer_extra_body,
)
from experiments.utils.token_counter import count_tokens, count_words
from systems.core.story_gen import LLAMACPP_NO_THINKING

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_text_record(
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


def _stego_path(output_dir: Path, system: str) -> Path:
    return output_dir / f"{system}_stego.jsonl"


# ---------------------------------------------------------------------------
# Configuration: generator, synthesizer, sampling (StorySlot / LitReview)
# ---------------------------------------------------------------------------

CONFIG_FLAGS = (
    "synth_model",
    "synth_provider",
    "generator_model",
    "generator_provider",
)
# Fields that define a configuration when checking a resume against stored
# records. Endpoint URLs are left out: a moved server is the same configuration.
_RESUME_KEYS = (
    "generator_model",
    "n_slots",
    "slot_margin",
    "synth_model",
    "synth_temperature",
    "synth_top_p",
    "synth_extra_body",
    "decoder_model",
)


def resolve_config(system: str, args: argparse.Namespace) -> dict:
    """The default configuration with every flag the caller set applied.

    Sampling is not a configuration axis: every writer runs at the default
    temperature/top_p, except one that takes only its provider's default
    sampling, which gets None for both."""
    config = default_config(system)
    for key in CONFIG_FLAGS:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if default_sampling_only(config["synth_provider"], config["synth_model"]):
        config["synth_temperature"] = config["synth_top_p"] = None
    return config


def config_system_kwargs(system: str, config: dict, generator_extra_body) -> dict:
    """make_story / make_litreview keyword arguments for a configuration.
    The writer's reasoning is fixed per model (see writer_extra_body)."""
    kwargs = {
        "synth_client": make_client(config["synth_provider"]),
        "synth_model": config["synth_model"],
        "synth_temperature": config["synth_temperature"],
        "synth_top_p": config["synth_top_p"],
        "synth_extra_body": writer_extra_body(
            config["synth_provider"], config["synth_model"]
        ),
    }
    if system == "story":
        kwargs["generator_model"] = config["generator_model"]
        kwargs["generator_extra_body"] = generator_extra_body
    return kwargs


def _check_resume_config(records_map: dict[str, dict], current: dict) -> None:
    """Refuse to resume into stego records generated under another configuration.

    Records written before configurations were recorded carry no config; they
    are the default configuration, and a non-default one never shares their
    subdir (it gets a config_tag suffix). A key a record predates is skipped.
    """
    for rid, record in records_map.items():
        if record.get("text_type") != "stego":
            continue
        stored = (record.get("metadata") or {}).get("config")
        if stored is None:
            continue
        diff = {
            k: (stored.get(k), current.get(k))
            for k in _RESUME_KEYS
            if k in stored and stored[k] != current.get(k)
        }
        if diff:
            raise SystemExit(
                f"{rid} was generated under a different configuration "
                f"(stored, requested): {diff}. Use a different --subdir."
            )


# ---------------------------------------------------------------------------
# StorySlot generation
# ---------------------------------------------------------------------------


def generate_story(
    client,
    generator_client,
    prompts: list[dict],
    messages: dict,
    output_dir: Path,
    n_slots: int = 20,
    slot_margin: int = 0,
    system_kwargs: dict | None = None,
):
    """Generate one StorySlot stegotext per prompt.

    ``system_kwargs`` (see config_system_kwargs) selects a non-default
    generator/synthesizer configuration.
    """
    stego_path = _stego_path(output_dir, "story")
    system = make_story(
        client,
        generator_client,
        n_slots=n_slots,
        slot_margin=slot_margin,
        **(system_kwargs or {}),
    )
    records_map = load_records_map(stego_path)
    completed = set(records_map)
    _check_resume_config(records_map, system.generation_config())

    stego_msgs = messages["stego_messages"]
    n_prompts = len(prompts)
    failures_path = output_dir / "story_failures.jsonl"

    log.info(f"StorySlot: {n_prompts} prompts, {len(completed)} records already done")

    for p_idx, prompt_data in enumerate(prompts):
        premise = prompt_data["premise"]
        log.info(f"StorySlot prompt {p_idx + 1}/{n_prompts}: {premise[:60]}...")

        s_rid = make_record_id("story", "stego", p_idx)
        if s_rid in completed:
            log.info(f"  Skip {s_rid} (exists)")
            continue
        msg_bits = stego_msgs[p_idx]
        try:
            text = system.hide_message(msg_bits, premise)
        except ValueError as e:
            # G returned fewer slots than the frame needs for this premise.
            # Log and skip, as LitReview does, rather than end the run.
            log.warning(f"  SKIP {s_rid} encode failure: {e}")
            append_jsonl(
                failures_path,
                {
                    "id": s_rid,
                    "stage": "stego",
                    "prompt_idx": p_idx,
                    "premise": premise,
                    "message_bits": msg_bits,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            continue
        stego_record = make_text_record(
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
        append_jsonl(stego_path, stego_record)
        log.info(f"  Generated {s_rid} ({stego_record['word_count']} words)")


# ---------------------------------------------------------------------------
# LitReview generation
# ---------------------------------------------------------------------------


def generate_litreview(
    client,
    corpus_indices: list[int],
    messages: dict,
    output_dir: Path,
    system_kwargs: dict | None = None,
):
    """Generate one LitReview stegotext per prompt.

    ``system_kwargs`` (see config_system_kwargs) selects a non-default
    synthesizer configuration.
    """
    stego_path = _stego_path(output_dir, "litreview")
    system = make_litreview(client, **(system_kwargs or {}))
    records_map = load_records_map(stego_path)
    completed = set(records_map)
    _check_resume_config(records_map, system.generation_config())

    stego_msgs = messages["stego_messages"]
    n_prompts = len(corpus_indices)

    log.info(f"LitReview: {n_prompts} prompts, {len(completed)} records already done")

    failures_path = output_dir / "litreview_failures.jsonl"

    for p_idx, corpus_idx in enumerate(corpus_indices):
        corpus_idx = int(corpus_idx)
        paper = system.corpus[corpus_idx]
        paper_title = paper["title"]
        log.info(
            f"LitReview prompt {p_idx + 1}/{n_prompts}: [{corpus_idx}] {paper_title[:60]}..."
        )

        s_rid = make_record_id("litreview", "stego", p_idx)
        if s_rid in completed:
            log.info(f"  Skip {s_rid} (exists)")
            continue
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
        stego_record = make_text_record(
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
        append_jsonl(stego_path, stego_record)
        log.info(f"  Generated {s_rid} ({stego_record['word_count']} words)")


# ---------------------------------------------------------------------------
# Token-level baseline (Discop) — in-house comparison system
# ---------------------------------------------------------------------------

# Mean stego length of the semantic systems at the capacities they are compared
# at (story 559-579w, litreview 576-622w across cap14-18). Length-matched
# Length-matched Discop runs target this so both sides hand the attacker a
# comparable cover, instead of the 3-4 word fragment it emits at its native rate.
BASELINE_LM_TARGET_WORDS = 575

# GPT-2's positional limit. Not a generation limit: both backends crop the KV
# cache to MAX_CONTEXT_LENGTH and keep going, so stego texts run well past it.
# Kept as documentation of where the attention window ends.
GPT2_CONTEXT_LIMIT = 1024


def _make_baseline_lm(
    system_name: str,
    repetitions: int,
    target_words: int,
    syncpool: bool = True,
):
    """Build the Discop baseline system at a given repetition rate.

    Discop's token cap is sized from the word target (GPT-2 runs ~1.4 tokens per
    word), with generous headroom. It is deliberately *not* clamped to 1024:
    that is GPT-2's positional limit, not a generation limit — the backend crops
    the KV cache (``_limit_past``) and generates past it. Clamping here truncated
    Discop mid-payload instead, producing records whose trailing bits were never
    embedded. ``DiscopSystem.hide_message`` now raises rather than emit one.
    """
    if system_name == "discop":
        # Generous headroom, not a tight estimate. Bits-per-token swings hard
        # across prompts — under the old 1024 clamp one cap14 record embedded
        # only 1792 of 3682 bits, i.e. it needed ~2x the budgeted length. Since
        # encoding stops the moment the payload is in, a loose cap costs nothing
        # on a typical document and only bounds a runaway.
        max_length = int(target_words * 1.7) * 4 + 512
        return make_discop(
            repetitions=repetitions, max_length=max_length, syncpool=syncpool
        )
    raise ValueError(f"unknown token-level baseline {system_name!r}")


def _warn_if_truncated(record_id: str, record: dict) -> None:
    """Flag stego records whose text channel cannot decode cleanly.

    Payload truncation is now an error at generation time (DiscopSystem raises),
    so what is left to warn about here is the BPE round trip.

    ``token_ids -> text -> token_ids`` is not the identity: GPT-2's tokenizer is
    greedy, so a pair the sampler emitted separately (``[" but", "tons"]``)
    comes back merged (``[" buttons"]``). From that token on, a text-channel
    decode reads a different sequence than the encoder wrote and every later bit
    is chance. The token channel (``metadata['token_ids']``) is unaffected, so
    this is a property of the transport, not of the record — but it is the
    reason a clean text-channel decode is below 1.0, and it should be visible at
    generation time rather than inferred from a bad recovery number later.
    """
    meta = record.get("metadata") or {}
    if meta.get("syncpool"):
        # SyncPool decodes by walking the stegotext's bytes, never by
        # re-tokenizing, so an inexact BPE round trip costs it nothing. That is
        # the entire point of turning it on, and warning here would flag most
        # records for a condition that no longer has consequences.
        return
    if meta.get("bpe_roundtrip_exact") is False:
        log.warning(
            f"  {record_id}: BPE round trip is not exact ({meta.get('n_tokens')} tokens emitted) — "
            f"the text channel will desync mid-stream. The token channel is unaffected."
        )


def _warn_if_off_target(record_id: str, record: dict, target_words: int) -> None:
    """Flag a length-matched record that missed the length it was matched to.

    The quiet form of the degeneration that makes `hide_message` raise. GPT-2
    slides into a repetition loop, the embedding rate collapses, and the encoder
    needs far more tokens to place the same payload — so the document survives
    but comes out much longer than `target_words`. That silently breaks the one
    property the length-matched configuration exists to provide (baseline and
    semantic systems handing the attacker comparable covers), and nothing
    downstream would notice.
    """
    words = record.get("word_count") or 0
    if not target_words:
        return
    ratio = words / target_words
    if ratio > 1.5 or ratio < 0.5:
        meta = record.get("metadata") or {}
        log.warning(
            f"  {record_id}: {words} words vs target {target_words} "
            f"({ratio:.1f}x) — length matching is broken for this record"
            + (
                f"; {meta['n_singleton_steps']} zero-entropy steps of "
                f"{meta.get('n_tokens')}"
                if meta.get("n_singleton_steps")
                else ""
            )
        )


def _odd(r: int) -> int:
    """Round a repetition rate up to the nearest odd value.

    An even rate leaves ties in the majority vote, which the decoder must break
    toward 0 — a half-vote bias against every message bit whose true value is 1.
    Rounding up costs at most one extra copy and removes the bias entirely.
    """
    return r if r % 2 else r + 1


def calibrate_repetitions(
    system_name: str,
    capacity: int,
    target_words: int,
    prompts: list[dict],
    n_pilot: int = 5,
    max_rounds: int = 3,
    syncpool: bool = True,
) -> int:
    """Pick the repetition rate whose mean stego length hits `target_words`.

    Discop emits ~5 bits/word natively, so a 16-bit payload lands in a
    3-4 word fragment — far too short for a paraphrase attack to be a meaningful
    operation on it. Inflating the payload with a repetition code is what buys a
    cover of comparable length to the semantic systems while holding the *message*
    fixed, so the baseline is handed both the same payload and the same cover
    budget rather than a degenerate one.

    Bits/word is not perfectly flat in length, so this re-measures at the
    estimated rate and refines, rather than extrapolating once from r=1.

    It must be re-run whenever `syncpool` or the local LM changes: both move
    bits/word by a large factor (SyncPool spends no payload on choices within an
    ambiguity pool), so an `r` calibrated under one setting length-matches
    nothing under another.
    """
    rng = np.random.default_rng(1234 + capacity)
    pilot = prompts[:n_pilot]
    r = 1
    for round_idx in range(max_rounds):
        system = _make_baseline_lm(system_name, r, target_words, syncpool=syncpool)
        words = []
        degenerate = 0
        for p in pilot:
            bits = rng.integers(0, 2, size=capacity).tolist()
            try:
                words.append(len(system.hide_message(bits, p["seed"]).split()))
            except ValueError as exc:
                # A pilot whose generation degenerated (see the exhaustion hint
                # in DiscopSystem) carries no usable rate estimate,
                # and it must not abort the run: this is a calibration probe, and
                # one bad trajectory out of five used to kill a multi-hour job.
                #
                # Drop it rather than folding it in. Its stego is enormous for
                # the bits it holds, so including it would *lower* measured
                # bits/word, which *raises* the next r, which makes the next
                # round's documents longer and more likely to degenerate still —
                # the estimator would chase its own tail.
                degenerate += 1
                log.warning(
                    f"  calibration round {round_idx + 1}: pilot "
                    f"{p['seed'][:50]!r} degenerated at r={r}; excluded from the "
                    f"rate estimate. {exc}"
                )
        if not words:
            raise RuntimeError(
                f"{system_name}: every calibration pilot degenerated at r={r}. "
                f"The length target ({target_words} words) is beyond what this "
                f"model sustains without falling into a repetition loop — lower "
                f"--target-words, or pin a rate with --repetitions."
            )
        mean_words = sum(words) / len(words)
        bits_per_word = (capacity * r) / mean_words
        next_r = _odd(max(1, round(target_words * bits_per_word / capacity)))
        log.info(
            f"  calibration round {round_idx + 1}: r={r} -> {mean_words:.0f} words "
            f"({bits_per_word:.2f} bits/word), target={target_words} => r={next_r}"
            + (f" [{degenerate}/{len(pilot)} pilots excluded]" if degenerate else "")
        )
        if next_r == r:
            break
        r = next_r

    log.info(f"{system_name}: calibrated repetitions r={r} for ~{target_words} words")
    return r


def generate_baseline_lm(
    system_name: str,
    prompts: list[dict],
    messages: dict,
    output_dir: Path,
    repetitions: int = 1,
    target_words: int = BASELINE_LM_TARGET_WORDS,
    syncpool: bool = True,
):
    """Generate one Discop stegotext per prompt.

    The token-level baseline encodes over a local GPT-2 (no API call); the
    prompt ``seed`` doubles as the LM generation context and is saved in
    ``system_state['context']`` so Phase 4 can re-run decoding.

    ``repetitions`` is the repetition-code rate (see `calibrate_repetitions`). It
    is written to ``system_state`` because the decode-side system is built at the
    factory default and has to be restored to the rate the record was encoded at.
    """
    system = _make_baseline_lm(
        system_name,
        repetitions,
        target_words,
        syncpool=syncpool,
    )

    stego_path = _stego_path(output_dir, system_name)
    records_map = load_records_map(stego_path)
    completed = set(records_map)

    stego_msgs = messages["stego_messages"]
    n_prompts = len(prompts)
    degenerate_prompts: list[int] = []

    log.info(
        f"{system_name}: {n_prompts} prompts, {len(completed)} records already done"
    )

    for p_idx, prompt_data in enumerate(prompts):
        seed = prompt_data["seed"]
        log.info(f"{system_name} prompt {p_idx + 1}/{n_prompts}: {seed[:60]}...")

        s_rid = make_record_id(system_name, "stego", p_idx)
        if s_rid in completed:
            log.info(f"  Skip {s_rid} (exists)")
            continue
        msg_bits = stego_msgs[p_idx]
        try:
            text = system.hide_message(msg_bits, seed)
        except ValueError as exc:
            # A degenerate trajectory (GPT-2 in a repetition loop, zero
            # embedding rate) is a property of this prompt/payload draw, not
            # of the run. Skipping costs one document; aborting costs the
            # hours already spent. Counted and reported at the end so the
            # shortfall is never silent.
            degenerate_prompts.append(p_idx)
            log.warning(f"  SKIP {s_rid}: {exc}")
            continue
        stego_record = make_text_record(
            record_id=s_rid,
            system=system_name,
            text_type="stego",
            prompt_idx=p_idx,
            prompt=seed,
            text=text,
            message_bits=msg_bits,
            system_state={
                "context": system._context,
                "error_encoded_length": system._error_encoded_length,
                "repetitions": repetitions,
                "interleave": getattr(system.ecc, "interleave", False),
                # Decoding a SyncPool stream without SyncPool (or the
                # reverse) yields chance, so the setting has to travel with
                # the record just as the ECC layout does.
                "syncpool": getattr(system, "syncpool", False),
            },
            metadata=system._last_metadata,
        )
        append_jsonl(stego_path, stego_record)
        _warn_if_truncated(s_rid, stego_record)
        _warn_if_off_target(s_rid, stego_record, target_words)
        log.info(f"  Generated {s_rid} ({stego_record['word_count']} words)")

    if degenerate_prompts:
        log.warning(
            f"{system_name}: {len(degenerate_prompts)} of {n_prompts} prompt(s) "
            f"produced no stego record because generation degenerated: "
            f"{degenerate_prompts}. Phase 3 takes the first 30 stegos by "
            f"prompt_idx, so raise --limit if this leaves you short."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# The token-level baseline runs outside the default "all" set; invoke it
# explicitly (it uses a local GPT-2 and its own capacity/messages).
BASELINE_LM_SYSTEMS = ("discop",)
BASELINE_LM_DEFAULT_CAPACITY = 16


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Text Generation")
    parser.add_argument(
        "--system",
        choices=[
            "story",
            "litreview",
            "discop",
            "all",
        ],
        default="all",
        help=(
            "Which system(s) to generate texts for. 'discop' is the "
            "in-house token-level baseline and is not included in 'all'."
        ),
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
        "--litreview-indices",
        default="litreview_indices.json",
        help=(
            "LitReview input list under prompts/ (default: the camera-ready one, "
            "papers with >= 60 usable references). The payload grid uses "
            "litreview_indices_min80.json (see expand_prompts --min-refs)."
        ),
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
        "--track",
        choices=TRACKS,
        default=None,
        help=(
            "Write the cell under phase1_texts/{track}/: detection texts "
            "(steganalysis, quality) and recovery texts (attacks, decoding) are "
            "generated separately. Default: no track (the camera-ready layout)."
        ),
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help=(
            "Override message-bit count for the chosen system. "
            "Requires --system != all. Auto-sets --subdir to '{system}_cap{N}' "
            "unless --subdir is given explicitly. "
            "Messages are regenerated inline via np.default_rng(42 + capacity) and "
            "written to {output_dir}/messages.json for traceability."
        ),
    )
    parser.add_argument(
        "--n-slots",
        type=int,
        default=None,
        help=(
            "Story only: number of plot slots, each carrying one bit. Defaults to "
            "--capacity (every slot carries a bit), or 20 without --capacity."
        ),
    )
    parser.add_argument(
        "--slot-margin",
        type=int,
        default=None,
        help=(
            "Story only: extra slots requested from G beyond --n-slots; the first "
            "n_slots are kept. Defaults to 2 with --capacity, 0 without."
        ),
    )
    parser.add_argument(
        "--length-matched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Discop only (default: on). Inflate the payload with a repetition code "
            "so the stego text reaches --target-words, matching the semantic systems' cover "
            "length at the same payload. At the native rate these systems emit 3-4 word "
            "fragments, which no paraphrase attack can meaningfully act on. "
            "--no-length-matched reproduces the native-rate reference condition."
        ),
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=BASELINE_LM_TARGET_WORDS,
        help=(
            f"Length-matched target stego length (default: {BASELINE_LM_TARGET_WORDS}, the "
            "mean of the semantic systems at cap14-18). Auto-sets --subdir to "
            "'{system}_cap{N}_len{T}'."
        ),
    )
    parser.add_argument(
        "--syncpool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Discop only (default: on). Eliminate segmentation ambiguity "
            "(Qi et al., arXiv:2403.17524) so a clean text-channel decode is exact. "
            "Without it the decoder re-tokenizes the stegotext into a different "
            "sequence than the encoder wrote and desyncs with no attacker present, "
            "which at length-matched repetition rates is fatal rather than merely "
            "costly. Auto-appends '_sp' to the subdir, since SyncPool and plain "
            "records are not interchangeable and must not share a checkpoint."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help=(
            "Length-matched repetition rate. Prefer an odd value: an even rate leaves "
            "ties that the majority vote breaks toward 0. "
            "Skips calibration when set; otherwise the rate "
            "is measured from a 5-prompt pilot against --target-words."
        ),
    )
    config_group = parser.add_argument_group(
        "configuration (story/litreview only)",
        "Generator and synthesizer. Unset flags keep the configuration "
        "every existing result was generated with (GPT-4.1 synthesizer via OpenAI, "
        "local LOCAL_MODEL generator). Sampling is fixed: top_p 0.7, T 0.7 for "
        "story / 0 for litreview, or the provider default for a writer that "
        "accepts no other (gpt-6-sol). "
        "A non-default configuration writes to a subdir suffixed with its tag.",
    )
    config_group.add_argument("--synth-model", default=None, help="Synthesizer model.")
    config_group.add_argument(
        "--synth-provider",
        choices=PROVIDERS,
        default=None,
        help="API serving --synth-model.",
    )
    config_group.add_argument(
        "--generator-model", default=None, help="Story only: generator (G) model."
    )
    config_group.add_argument(
        "--generator-provider",
        choices=PROVIDERS,
        default=None,
        help="Story only: API serving --generator-model.",
    )
    config_group.add_argument(
        "--generator-extra-body",
        type=json.loads,
        default=None,
        help=(
            "Story only: JSON request body sent with generator calls (provider-specific; "
            "'null' sends none). Default: the llama.cpp no-thinking body for the local "
            "provider, none otherwise."
        ),
    )
    args = parser.parse_args()

    # --- Configuration: resolve flags; a non-default one gets its own subdir ---
    config = None
    set_flags = [k for k in CONFIG_FLAGS if getattr(args, k) is not None]
    if args.generator_extra_body is not None:
        set_flags.append("generator_extra_body")
    if set_flags:
        if args.system not in CONFIG_SYSTEMS:
            parser.error(
                f"--{set_flags[0].replace('_', '-')} needs --system story or litreview."
            )
        generator_flags = [f for f in set_flags if f.startswith("generator_")]
        if args.system == "litreview" and generator_flags:
            parser.error("LitReview has no generator; drop the --generator-* flags.")
        config = resolve_config(args.system, args)
        if config == default_config(args.system):
            config = None
    hosted_generator = config is not None and config["generator_provider"] not in (
        None,
        "local",
    )
    if args.generator_extra_body is not None and not hosted_generator:
        parser.error(
            "--generator-extra-body is for a hosted --generator-provider; the local "
            "server always gets the llama.cpp no-thinking body."
        )
    generator_extra_body = (
        args.generator_extra_body if hosted_generator else LLAMACPP_NO_THINKING
    )

    # Token-level baselines default to a native payload if none is given, so
    # their messages come from the inline-capacity path (they have no entry in
    # the shared messages.json).
    if args.system in BASELINE_LM_SYSTEMS and args.capacity is None:
        args.capacity = BASELINE_LM_DEFAULT_CAPACITY
        log.info(
            f"{args.system}: no --capacity given, defaulting to "
            f"{BASELINE_LM_DEFAULT_CAPACITY}-bit payload"
        )

    # --- Capacity handling: requires a specific system; auto-subdir; inline messages ---
    if args.capacity is not None:
        if args.system == "all":
            parser.error(
                "--capacity requires --system to be one of story/litreview/discop (not 'all')."
            )
        if args.subdir == "recovery_test":
            args.subdir = f"{args.system}_cap{args.capacity}"
            # Length-matched baseline runs get their own dir so they sit alongside
            # the native-rate ones rather than overwriting them — the two are
            # different experimental conditions and the paper reports both.
            if args.system in BASELINE_LM_SYSTEMS and args.length_matched:
                args.subdir += f"_len{args.target_words}"
            if args.system in BASELINE_LM_SYSTEMS and args.syncpool:
                args.subdir += "_sp"
            log.info(f"--capacity set: defaulting --subdir to {args.subdir!r}")

    if config is not None:
        tag = config_tag(args.system, config)
        args.subdir = f"{args.subdir}_{tag}" if args.subdir else tag
        log.info(f"Configuration {config}; subdir {args.subdir!r}")
    if args.track is not None:
        args.subdir = f"{args.track}/{args.subdir}" if args.subdir else args.track
        log.info(f"Track {args.track}; subdir {args.subdir!r}")

    prompts_dir = args.data_dir / "prompts"
    output_dir = args.data_dir / "phase1_texts"
    if args.subdir:
        output_dir = output_dir / args.subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir}")
    if args.limit is not None:
        log.info(f"Limiting to first {args.limit} prompt(s) per system")

    # --- Messages: inline regen for capacity variant, else load shared messages.json ---
    if args.capacity is not None:
        # Deterministic seed: same prompt set used across all variants, but
        # bits differ per capacity (different num_bits means different draws).
        rng = np.random.default_rng(42 + args.capacity)
        n_prompts_msg = 300  # matches expand_prompts.TARGET_MESSAGES
        stego = rng.integers(0, 2, size=(n_prompts_msg, args.capacity)).tolist()
        all_messages = {
            args.system: {
                "num_bits": args.capacity,
                "stego_messages": stego,
            }
        }
        # Persist for traceability + Phase 4 sanity checks.
        msgs_path = output_dir / "messages.json"
        msgs_path.write_text(
            json.dumps(
                {
                    "seed": 42 + args.capacity,
                    "rng": "numpy.default_rng",
                    "system": args.system,
                    "capacity": args.capacity,
                    **all_messages,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )
        log.info(
            f"Wrote variant messages.json ({args.capacity} bits, {n_prompts_msg} prompts) to {msgs_path}"
        )
    else:
        with open(prompts_dir / "messages.json") as f:
            all_messages = json.load(f)

    client, generator_client = make_clients()
    system_kwargs = None
    if config is not None:
        system_kwargs = config_system_kwargs(args.system, config, generator_extra_body)
        if args.system == "story":
            generator_client = make_client(config["generator_provider"])

    if args.system in ("story", "all"):
        with open(prompts_dir / "story_prompts.json") as f:
            prompts = json.load(f)["prompts"]
        if args.limit is not None:
            prompts = prompts[: args.limit]
        # With --capacity every slot carries a bit (n_slots = F), and G is asked
        # for F + 2 because it miscounts long lists. Without it: the original
        # 20 slots requested as-is.
        if args.n_slots is not None:
            story_n_slots = args.n_slots
        else:
            story_n_slots = args.capacity if args.capacity is not None else 20
        if args.slot_margin is not None:
            story_slot_margin = args.slot_margin
        else:
            story_slot_margin = 2 if args.capacity is not None else 0
        generate_story(
            client,
            generator_client,
            prompts,
            all_messages["story"],
            output_dir,
            n_slots=story_n_slots,
            slot_margin=story_slot_margin,
            system_kwargs=system_kwargs,
        )

    if args.system in ("litreview", "all"):
        with open(prompts_dir / args.litreview_indices) as f:
            indices_data = json.load(f)
        indices = indices_data["indices"]
        if args.limit is not None:
            indices = indices[: args.limit]
        generate_litreview(
            client,
            indices,
            all_messages["litreview"],
            output_dir,
            system_kwargs=system_kwargs,
        )

    if args.system in BASELINE_LM_SYSTEMS:
        with open(prompts_dir / f"{args.system}_prompts.json") as f:
            prompts = json.load(f)["prompts"]
        if args.limit is not None:
            prompts = prompts[: args.limit]

        if args.syncpool:
            log.info(f"{args.system}: SyncPool on")
        else:
            log.warning(
                f"{args.system}: --no-syncpool — the clean text channel will desync "
                f"mid-stream with no attacker present. Reference condition only."
            )

        repetitions = 1
        if args.length_matched:
            repetitions = args.repetitions or calibrate_repetitions(
                args.system,
                args.capacity,
                args.target_words,
                prompts,
                syncpool=args.syncpool,
            )
            log.info(
                f"{args.system}: length-matched at r={repetitions} "
                f"(~{args.target_words} words, {args.capacity}-bit payload)"
            )
        else:
            log.warning(
                f"{args.system}: --no-length-matched — running at the native rate, which "
                f"yields 3-4 word stego texts at {args.capacity} bits. Paraphrase attacks on "
                f"texts that short are not a meaningful operation; this is a reference "
                f"condition, not the headline comparison."
            )

        generate_baseline_lm(
            args.system,
            prompts,
            all_messages[args.system],
            output_dir,
            repetitions=repetitions,
            target_words=args.target_words,
            syncpool=args.syncpool,
        )

    log.info("Phase 1 generation complete.")


if __name__ == "__main__":
    main()
