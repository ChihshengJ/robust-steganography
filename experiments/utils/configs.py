"""Generation configurations: (generator G, synthesizer, sampling).

Shared by phase 1 (which writes them) and the steganalysis code (which groups
texts by them). Prompts are fixed; a configuration is only which models run
and how they sample.
"""

from __future__ import annotations

from experiments.utils.io import model_slug
from experiments.utils.system_factory import LOCAL_MODEL, provider_for_base_url

CONFIG_SYSTEMS = ("story", "litreview")


def default_config(system: str) -> dict:
    """The configuration every existing result was generated with."""
    return {
        "synth_model": "gpt-4.1",
        "synth_provider": "openai",
        "synth_temperature": 0.7 if system == "story" else 0.0,
        "synth_top_p": 0.7,
        "generator_model": LOCAL_MODEL if system == "story" else None,
        "generator_provider": "local" if system == "story" else None,
    }


def config_tag(system: str, config: dict) -> str:
    """Subdir suffix naming a configuration, e.g.
    'syn-deepseek-v4-flash_t1_p0.95_gen-qwen3.5-9b'."""
    tag = (
        f"syn-{model_slug(config['synth_model'])}"
        f"_t{config['synth_temperature']:g}_p{config['synth_top_p']:g}"
    )
    if system == "story":
        tag += f"_gen-{model_slug(config['generator_model'])}"
    return tag


def stego_config(record: dict) -> dict:
    """The configuration a stego record was generated under, in the shape of
    ``generation_config()`` plus the synthesizer's provider.

    Records written before configurations were recorded carry none; they are
    the default configuration.
    """
    stored = (record.get("metadata") or {}).get("config")
    if stored is None:
        default = default_config(record["system"])
        stored = {
            "generator_model": default["generator_model"],
            "synth_model": default["synth_model"],
            "synth_base_url": None,
            "synth_temperature": default["synth_temperature"],
            "synth_top_p": default["synth_top_p"],
        }
    return {
        **stored,
        "synth_provider": provider_for_base_url(stored.get("synth_base_url")),
    }
