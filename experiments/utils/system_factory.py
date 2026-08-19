"""Factory functions for creating steganography systems and restoring state."""

from __future__ import annotations

import os

import openai

from systems import (
    CORPORATE_MONOLOGUE,
    DiscopLM,
    DiscopSystem,
    LitReviewSystem,
    MeteorLM,
    MeteorSystem,
    RandomProjectionHash,
    RepetitionCode,
    SentenceStegSystem,
    StorySystem,
    TopicQASystem,
    BypassEncoder
)
from systems.core.litreview import load_corpus
from systems.paths import litreview_references

# Local (llama.cpp / OpenAI-compatible) server used for deterministic subtopic
# and slot generation. Overridable from the environment so the same scripts run
# against a different host, port, or GGUF model without editing source. The
# defaults match experiments/serve_local_model.sh (PORT=8080); LOCAL_MODEL must
# equal the basename of the GGUF you serve (the alias llama-server reports on
# /v1/models). `import systems` has already loaded .env by this point, so values
# defined there are visible here.
LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "Qwen3.5-4B-UD-Q8_K_XL.gguf")


def make_clients(
    local_base_url: str | None = None,
) -> tuple[openai.OpenAI, openai.OpenAI]:
    """Create the OpenAI API client and the local llama.cpp client.

    The remote client honours OpenAI's own env vars (``OPENAI_API_KEY``, and
    ``OPENAI_BASE_URL`` if you front it with a proxy). The local client points
    at ``local_base_url`` (default: ``LOCAL_BASE_URL`` / ``$LOCAL_BASE_URL``).
    """
    client = openai.OpenAI()
    local_client = openai.OpenAI(
        base_url=local_base_url or LOCAL_BASE_URL,
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


# Local causal LM backing the token-level baselines (Meteor, Discop). GPT-2 is
# the paper-faithful Meteor/Discop channel; override with $BASELINE_MODEL.
BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "gpt2")

# Loading a HF model is expensive; cache the LM holders by (backend, model) so
# repeated factory calls within a run reuse one in-memory model.
_METEOR_LM_CACHE: dict[str, MeteorLM] = {}
_DISCOP_LM_CACHE: dict[str, DiscopLM] = {}


def make_meteor(
    model_name: str | None = None,
    key: str = "default",
    repetitions: int = 1,
) -> MeteorSystem:
    """Meteor (Kaptchuk et al., CCS 2021) token-level baseline over local GPT-2.

    In-house baseline only. `key` is the shared symmetric passphrase; the seed
    context is supplied per-message via `hide_message(seed=...)`.

    `repetitions` sets the repetition-code rate. At r=1 the system runs at its
    native ~5 bits/word, which yields 3-4 word stego texts at the payloads the
    semantic systems use — too short for a paraphrase attack to be meaningful.
    Length-matched runs raise r so the same payload fills a cover comparable in
    length to the semantic systems' (~575 words); see phase1_generate's
    --length-matched.
    """
    model_name = model_name or BASELINE_MODEL
    lm = _METEOR_LM_CACHE.get(model_name)
    if lm is None:
        lm = MeteorLM(model_name)
        _METEOR_LM_CACHE[model_name] = lm
    return MeteorSystem(
        lm,
        error_correction=RepetitionCode(repetitions),
        encoder=BypassEncoder(),
        key=key,
    )


def make_discop(
    model_name: str | None = None,
    key: str = "default",
    repetitions: int = 1,
    max_length: int = 512,
) -> DiscopSystem:
    """Discop (Ding et al., S&P 2023) token-level baseline over local GPT-2.

    In-house baseline only. `key` is the shared symmetric passphrase (Discop's
    sampling seed); the context is supplied per-message via `hide_message`.

    `repetitions` behaves as in `make_meteor`. `max_length` caps generated
    tokens and must be raised alongside it — the 512 default truncates well
    below a 575-word length-matched cover, which would silently drop payload.
    """
    model_name = model_name or BASELINE_MODEL
    lm = _DISCOP_LM_CACHE.get(model_name)
    if lm is None:
        lm = DiscopLM(model_name)
        _DISCOP_LM_CACHE[model_name] = lm
    return DiscopSystem(
        lm,
        error_correction=RepetitionCode(repetitions),
        encoder=BypassEncoder(),
        key=key,
        max_length=max_length,
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
    if "context" in state_dict:
        # Meteor/Discop baselines: the generation context (seed string) that
        # decoding must re-run the local LM from.
        system._context = state_dict["context"]
    if "error_encoded_length" in state_dict:
        system._error_encoded_length = state_dict["error_encoded_length"]
    if "repetitions" in state_dict:
        # Length-matched Meteor/Discop encode at r > 1. The factory builds the
        # decode-side system at its default rate, so the record's rate has to
        # win here or RepetitionCode.decode would fold the wrong block size.
        system.ecc = RepetitionCode(int(state_dict["repetitions"]))
