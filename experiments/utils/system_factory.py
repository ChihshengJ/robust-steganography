"""Factory functions for creating steganography systems and restoring state."""

from __future__ import annotations

import os

import openai

from systems import (
    BypassEncoder,
    DiscopLM,
    DiscopSystem,
    LitReviewSystem,
    RepetitionCode,
    StorySystem,
)
from systems.core.litreview import load_corpus
from systems.core.story_gen import LLAMACPP_NO_THINKING
from systems.paths import litreview_references

# Local (llama.cpp / OpenAI-compatible) server used for deterministic slot
# generation. Overridable from the environment so the same scripts run
# against a different host, port, or GGUF model without editing source. The
# defaults match experiments/serve_local_model.sh (PORT=8080); LOCAL_MODEL must
# equal the basename of the GGUF you serve (the alias llama-server reports on
# /v1/models). `import systems` has already loaded .env by this point, so values
# defined there are visible here. The default is the pinned recovery G of the
# payload grid; recovery SG folder names embed it, so a wrong default points
# every grid stage at folders that do not exist.
LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "Qwen3.5-9B-UD-Q8_K_XL.gguf")


def make_clients(
    generator_base_url: str | None = None,
) -> tuple[openai.OpenAI, openai.OpenAI]:
    """Create the OpenAI API client and the generator client.

    The API client honours OpenAI's own env vars (``OPENAI_API_KEY``, and
    ``OPENAI_BASE_URL`` if you front it with a proxy). The generator client
    points at ``generator_base_url``, by default the local llama.cpp server
    (``LOCAL_BASE_URL`` / ``$LOCAL_BASE_URL``).
    """
    client = openai.OpenAI()
    generator_client = openai.OpenAI(
        base_url=generator_base_url or LOCAL_BASE_URL,
        api_key="unused",
    )
    return client, generator_client


# OpenRouter, OpenAI-compatible: every hosted model other than OpenAI's (the
# DeepSeek writer and decoder, the detection G models, the attacker, the
# judges). Any model other than the pinned G is treated as a public endpoint,
# so which upstream OpenRouter routes a call to is not pinned. Overridable like
# LOCAL_BASE_URL.
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def make_openrouter_client() -> openai.OpenAI:
    """Create an OpenAI-compatible client for OpenRouter (``OPENROUTER_API_KEY``)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (add it to .env).")
    return openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


PROVIDERS = ("openai", "openrouter", "local")


def make_client(provider: str) -> openai.OpenAI:
    """Client for a provider: 'openai', 'openrouter' (OPENROUTER_API_KEY), or
    'local' (the pinned llama.cpp server at LOCAL_BASE_URL)."""
    if provider == "openai":
        return openai.OpenAI()
    if provider == "openrouter":
        return make_openrouter_client()
    if provider == "local":
        return openai.OpenAI(base_url=LOCAL_BASE_URL, api_key="unused")
    raise ValueError(f"unknown provider {provider!r}; choose from {PROVIDERS}")


# OpenAI models that reason by default and take temperature/top_p only with
# reasoning_effort="none".
_OPENAI_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


def no_reasoning_body(provider: str, model: str) -> dict | None:
    """Request body that turns a hosted writer's reasoning off, or None.

    Synthesizers (and the normal generations that share their configuration)
    write without reasoning, as GPT-4.1 did: a reasoning model otherwise spends
    its whole length cap thinking (DeepSeek V4.1 Flash writes no story in
    3000 tokens), and OpenAI's refuse temperature/top_p unless reasoning is off.
    """
    if provider == "openrouter":
        return {"reasoning": {"enabled": False}}
    if provider == "openai" and model.startswith(_OPENAI_REASONING_PREFIXES):
        return {"reasoning_effort": "none"}
    return None


# Writers with a fixed reasoning setting (decided 09-27 after the SG F = 32
# pilot: with reasoning, both keep all 32 details; without it DeepSeek drops
# some). Any other writer runs with reasoning off (no_reasoning_body).
WRITER_REASONING = {
    "deepseek/deepseek-v4.1-flash": {"reasoning": {"enabled": True}},
    "gpt-6-sol": {"reasoning_effort": "low"},
}


def writer_extra_body(provider: str, model: str) -> dict | None:
    """Request body a synthesizer (and its normal generations) is sent."""
    if model in WRITER_REASONING:
        return WRITER_REASONING[model]
    return no_reasoning_body(provider, model)


# The decoder (REVISION_PLAN_ARR_OCT.md §1): one model for every task and
# configuration, chosen at decode time (phase4a --decoder-model).
DECODER_PROVIDER = "openrouter"
DECODER_MODEL = "deepseek/deepseek-v4.1-flash"


def decoder_extra_body(provider: str, model: str) -> dict | None:
    """Request body the decoder is sent: reasoning on through OpenRouter, sent
    explicitly although it is DeepSeek V4.1 Flash's default there (checked
    09-29), so a change of default cannot switch it off. Elsewhere, the
    provider default."""
    if provider == "openrouter":
        return {"reasoning": {"enabled": True}}
    return None


def default_sampling_only(provider: str, model: str) -> bool:
    """Whether the writer takes no temperature/top_p: an OpenAI reasoning
    model with reasoning on accepts only its default sampling."""
    body = writer_extra_body(provider, model) or {}
    return provider == "openai" and body.get("reasoning_effort", "none") != "none"


def provider_for_base_url(base_url: str | None) -> str:
    """The provider behind a base URL recorded in a generation config, so a
    stored config can be served again with ``make_client``. None is OpenAI:
    records from before base URLs were recorded all used it."""
    if base_url is None:
        return "openai"
    openai_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    urls = {
        "openai": openai_url,
        "openrouter": OPENROUTER_BASE_URL,
        "local": LOCAL_BASE_URL,
    }
    for provider, url in urls.items():
        if base_url.rstrip("/") == url.rstrip("/"):
            return provider
    raise ValueError(f"no provider serves {base_url!r}; known: {urls}")


def make_story(
    client: openai.OpenAI,
    generator_client: openai.OpenAI,
    n_slots: int = 16,
    *,
    slot_margin: int = 0,
    generator_model: str | None = None,
    generator_extra_body: dict | None = LLAMACPP_NO_THINKING,
    synth_client: openai.OpenAI | None = None,
    synth_model: str = "gpt-4.1",
    synth_temperature: float | None = 0.7,
    synth_top_p: float | None = 0.7,
    synth_extra_body: dict | None = None,
    decoder_client: openai.OpenAI | None = None,
    decoder_model: str = "gpt-4.1",
    decoder_extra_body: dict | None = None,
) -> StorySystem:
    """Create a StorySystem with standard experiment parameters.

    Capacity = n_slots bits (1 bit per slot ranking); G is asked for
    ``n_slots + slot_margin`` slots and the first ``n_slots`` are kept. The generator G runs on
    ``generator_client`` with ``generator_model`` (default LOCAL_MODEL). The
    synthesizer defaults to ``client``, and the decoder to ``client`` with
    GPT-4.1. The defaults are the configuration every existing result was
    generated with.
    """
    return StorySystem(
        decoder_client or client,
        error_correction=RepetitionCode(1),
        generator_client=generator_client,
        generator_model=generator_model or LOCAL_MODEL,
        n_slots=n_slots,
        slot_margin=slot_margin,
        synth_model=synth_model,
        decoder_model=decoder_model,
        key="default",
        encoder=BypassEncoder(),
        synth_temperature=synth_temperature,
        synth_client=synth_client or client,
        synth_top_p=synth_top_p,
        generator_extra_body=generator_extra_body,
        synth_extra_body=synth_extra_body,
        decoder_extra_body=decoder_extra_body,
    )


def make_litreview(
    client: openai.OpenAI,
    *,
    synth_client: openai.OpenAI | None = None,
    synth_model: str = "gpt-4.1",
    synth_temperature: float | None = 0.0,
    synth_top_p: float | None = 0.7,
    synth_extra_body: dict | None = None,
    decoder_client: openai.OpenAI | None = None,
    decoder_model: str = "gpt-4.1",
    decoder_extra_body: dict | None = None,
) -> LitReviewSystem:
    """Create a LitReviewSystem with corpus loaded.

    The synthesizer defaults to ``client``, and the citation extractor
    (decoder) to ``client`` with GPT-4.1. The defaults are the configuration
    every existing result was generated with.
    """
    corpus = load_corpus(*litreview_references())
    return LitReviewSystem(
        decoder_client or client,
        error_correction=RepetitionCode(1),
        corpus=corpus,
        model=decoder_model,
        encoder=BypassEncoder(),
        key="default",
        synth_client=synth_client or client,
        synth_model=synth_model,
        synth_temperature=synth_temperature,
        synth_top_p=synth_top_p,
        synth_extra_body=synth_extra_body,
        decoder_extra_body=decoder_extra_body,
    )


# Local causal LM backing the Discop baseline. Override with $BASELINE_MODEL;
# the length-matched runs set it to gpt2-medium, since gpt2-small degenerates
# into repetition loops well before the ~750 tokens those runs require
# (ECC_AND_RATE_PLAN.md §9.4).
BASELINE_MODEL = os.environ.get("BASELINE_MODEL", "gpt2")

# Loading a HF model is expensive; cache the LM holders by model so repeated
# factory calls within a run reuse one in-memory model.
_DISCOP_LM_CACHE: dict[str, DiscopLM] = {}


def make_discop(
    model_name: str | None = None,
    key: str = "default",
    repetitions: int = 1,
    max_length: int = 512,
    interleave: bool = True,
    syncpool: bool = True,
) -> DiscopSystem:
    """Discop (Ding et al., S&P 2023) token-level baseline over local GPT-2.

    In-house baseline only. `key` is the shared symmetric passphrase (Discop's
    sampling seed); the context is supplied per-message via `hide_message`.

    `repetitions` sets the repetition-code rate: at r=1 Discop runs at its
    native rate, which yields a 3-4 word stego text at the payloads the
    semantic systems use — too short for a paraphrase attack to be meaningful.
    Length-matched runs raise r so the same payload fills a comparable cover;
    see phase1_generate's --length-matched. `syncpool` turns on
    segmentation-ambiguity elimination (Qi et al. 2024), which Discop is the
    scheme they validated it on. `max_length` caps generated
    tokens and must be raised alongside both: the 512 default truncates well
    below a 575-word length-matched cover, and SyncPool lowers bits/token, so
    the same payload needs more of them.
    """
    model_name = model_name or BASELINE_MODEL
    lm = _DISCOP_LM_CACHE.get(model_name)
    if lm is None:
        lm = DiscopLM(model_name)
        _DISCOP_LM_CACHE[model_name] = lm
    return DiscopSystem(
        lm,
        error_correction=RepetitionCode(repetitions, interleave=interleave),
        encoder=BypassEncoder(),
        key=key,
        max_length=max_length,
        syncpool=syncpool,
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
        # Discop baseline: the generation context (seed string) that decoding
        # must re-run the local LM from.
        system._context = state_dict["context"]
    if "error_encoded_length" in state_dict:
        system._error_encoded_length = state_dict["error_encoded_length"]
    if "repetitions" in state_dict:
        # Length-matched Discop encodes at r > 1. The factory builds the
        # decode-side system at its default rate, so the record's rate has to
        # win here or RepetitionCode.decode would fold the wrong block size.
        #
        # The layout has to come from the record too: block and interleaved
        # streams are not interchangeable, and folding one as the other yields
        # chance. Records written before the interleaved layout existed have no
        # `interleave` key and are block-coded, so False is the right default.
        system.ecc = RepetitionCode(
            int(state_dict["repetitions"]),
            interleave=bool(state_dict.get("interleave", False)),
        )
    if hasattr(system, "syncpool"):
        # Discop: SyncPool changes both the emitted token sequence and
        # how the decoder walks it, so a stream encoded one way and decoded the
        # other yields chance. Records written before SyncPool existed have no
        # key and were encoded without it, so False is the right default — and
        # the factory default (on) must not leak into those decodes.
        system.syncpool = bool(state_dict.get("syncpool", False))
