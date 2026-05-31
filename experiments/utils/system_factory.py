"""Factory functions for creating steganography systems and restoring state."""

from __future__ import annotations

import openai

from systems import (
    CORPORATE_MONOLOGUE,
    LitReviewSystem,
    RandomProjectionHash,
    RepetitionCode,
    SentenceStegSystem,
    StorySystem,
    TopicQASystem,
    BypassEncoder
)
from systems.core.litreview import load_corpus
from systems.paths import litreview_references

LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"
LOCAL_MODEL = "Qwen3.5-4B-UD-Q8_K_XL.gguf"


def make_clients() -> tuple[openai.OpenAI, openai.OpenAI]:
    """Create the OpenAI API client and local Ollama client."""
    client = openai.OpenAI()
    local_client = openai.OpenAI(
        base_url=LOCAL_BASE_URL,
        api_key="unused",
    )
    return client, local_client


def make_topicqa(
    client: openai.OpenAI,
    local_client: openai.OpenAI,
    n_subtopics: int = 12,
    group_size: int = 2,
) -> TopicQASystem:
    """Create a TopicQASystem with standard experiment parameters.

    Capacity = n_subtopics // group_size * log2(group_size). For the default
    group_size=2, that's n_subtopics // 2 bits.
    """
    return TopicQASystem(
        client,
        error_correction=RepetitionCode(1),
        local_client=local_client,
        local_model=LOCAL_MODEL,
        n_subtopics=n_subtopics,
        group_size=group_size,
        response_model="gpt-4.1",
        decoder_model="gpt-4.1",
        key="default",
        encoder=BypassEncoder(),
        response_temperature=0.7,
    )


def make_story(
    client: openai.OpenAI,
    local_client: openai.OpenAI,
    n_slots: int = 16,
) -> StorySystem:
    """Create a StorySystem with standard experiment parameters.

    Capacity = n_slots bits (1 bit per slot ranking).
    """
    return StorySystem(
        client,
        error_correction=RepetitionCode(1),
        local_client=local_client,
        local_model=LOCAL_MODEL,
        n_slots=n_slots,
        response_model="gpt-4.1",
        decoder_model="gpt-4.1",
        key="default",
        encoder=BypassEncoder(),
        response_temperature=0.7,
    )


def make_litreview(client: openai.OpenAI) -> LitReviewSystem:
    """Create a LitReviewSystem with corpus loaded."""
    corpus = load_corpus(*litreview_references())
    return LitReviewSystem(
        client,
        error_correction=RepetitionCode(1),
        corpus=corpus,
        model="gpt-4.1",
        encoder=BypassEncoder(),
        key="default",
    )


def make_baseline(client: openai.OpenAI) -> SentenceStegSystem:
    """Baseline sentence-level steg (Bauer et al.): random-projection hash,
    1 bit/sentence, RepetitionCode(5), corporate-email generation prompt."""
    hash_fn = RandomProjectionHash(seed=108)
    return SentenceStegSystem(
        client,
        hash_function=hash_fn,
        error_correction=RepetitionCode(5),
        system_prompt=CORPORATE_MONOLOGUE,
        encoder=BypassEncoder(),
    )


def restore_system_state(system, state_dict: dict) -> None:
    """Restore internal state for decoding from a saved state dict.

    Works for any of the three systems — sets whichever attributes are
    present in the state dict. Keys match the experiment.md schema
    (non-underscored) and are translated to the `_`-prefixed private
    attrs on the system objects.
    """
    if "question" in state_dict:
        system._question = state_dict["question"]
    if "premise" in state_dict:
        system._premise = state_dict["premise"]
    if "error_encoded_length" in state_dict:
        system._error_encoded_length = state_dict["error_encoded_length"]
