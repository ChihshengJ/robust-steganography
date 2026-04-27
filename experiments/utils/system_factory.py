"""Factory functions for creating steganography systems and restoring state."""

from __future__ import annotations

import openai

from embeddings import LitReviewSystemV2, RepetitionCode, StorySystemV2, TopicQASystem
from embeddings.core.litreview_v2 import load_corpus
from measurements.utils import BypassEncoder

LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
LOCAL_MODEL = "Qwen3.5-4B-UD-Q8_K_XL.gguf"


def make_clients() -> tuple[openai.OpenAI, openai.OpenAI]:
    """Create the OpenAI API client and local Ollama client."""
    client = openai.OpenAI()
    local_client = openai.OpenAI(base_url=LOCAL_BASE_URL, api_key="unused")
    return client, local_client


def make_topicqa(client: openai.OpenAI, local_client: openai.OpenAI) -> TopicQASystem:
    """Create a TopicQASystem with standard experiment parameters."""
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


def make_story(client: openai.OpenAI, local_client: openai.OpenAI) -> StorySystemV2:
    """Create a StorySystemV2 with standard experiment parameters."""
    return StorySystemV2(
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


def make_litreview(client: openai.OpenAI) -> LitReviewSystemV2:
    """Create a LitReviewSystemV2 with corpus loaded."""
    corpus = load_corpus(
        "src/pca/litreview/references/corpus.jsonl",
        "src/pca/litreview/references/references.jsonl",
    )
    return LitReviewSystemV2(
        client,
        error_correction=RepetitionCode(1),
        corpus=corpus,
        model="gpt-4.1",
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
