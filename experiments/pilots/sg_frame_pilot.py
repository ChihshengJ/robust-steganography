"""Pilot: can StorySlot carry a full F-bit frame (one bit in every slot)?

Before generating the payload grid, check at the largest frame size F that G
returns enough distinct A/B slots and that the clean story decodes. Per premise:

1. Ask G for F + margin slots; keep the first F (as StorySystem does).
2. Encode a uniform F-bit message (the grid's rng, seed 42 + F) into a story.
3. Decode the unattacked story with the same slots.

Reports:
- how many slots G returned before truncation, and parse failures;
- near-duplicates: A ~ B within a slot, and slot names or alternatives repeated
  across slots, by token overlap and by embedding cosine (the pairs are listed
  so they can be checked by hand);
- story length in words;
- clean recovery of all F bits.

G and decoder run on OpenRouter (the 09-27 pilot runs used Together serverless),
the synthesizer on
--synth-provider; the decoder reasons, the synthesizer reasons only with
--synth-reasoning (an OpenAI reasoning writer then runs at its default
sampling, since it rejects temperature/top_p). G's determinism
is not under test here: the decoder is handed the encoder's slots.
--reuse-slots takes the slots from an earlier run's records instead of calling
G (serverless G is not deterministic), so two runs differ only in the writer.

Usage:
    python -m experiments.pilots.sg_frame_pilot --frame 32 --n-premises 20
    python -m experiments.pilots.sg_frame_pilot --frame 32 --synth-reasoning \
        --reuse-slots data/experiments/phase1_texts/pilot_sg_f32/pilot_records.jsonl \
        --tag reason
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openai

from experiments.utils.system_factory import (
    make_client,
    make_story,
    make_openrouter_client,
    no_reasoning_body,
)
from systems.config.story_prompts import SLOT_GENERATION_PROMPT
from systems.core.story_gen import LLAMACPP_NO_THINKING, _parse_slots
from systems.utils.new_text import llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the of in on at to for with by from and or its their his her into".split()
)


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


def near_duplicates(slots: list[dict], embed, jac: float, cos: float) -> dict:
    """Pairs that would make a slot ambiguous to decode."""
    within = [
        {"slot": i, "A": s["A"], "B": s["B"], "jaccard": round(jaccard(s["A"], s["B"]), 2)}
        for i, s in enumerate(slots)
        if jaccard(s["A"], s["B"]) >= jac
    ]
    names = [
        {"slots": [i, j], "names": [slots[i]["slot"], slots[j]["slot"]]}
        for i, j in itertools.combinations(range(len(slots)), 2)
        if jaccard(slots[i]["slot"], slots[j]["slot"]) >= jac
    ]
    # Alternatives of different slots: lexical and embedding similarity.
    alts = [(i, k, s[k]) for i, s in enumerate(slots) for k in ("A", "B")]
    vecs = embed([a[2] for a in alts])
    sims = vecs @ vecs.T
    across = []
    for x, y in itertools.combinations(range(len(alts)), 2):
        (i, ki, ti), (j, kj, tj) = alts[x], alts[y]
        if i == j:
            continue
        lj, cs = jaccard(ti, tj), float(sims[x, y])
        if lj >= jac or cs >= cos:
            across.append(
                {
                    "slots": [i, j],
                    "texts": [f"{ki}: {ti}", f"{kj}: {tj}"],
                    "jaccard": round(lj, 2),
                    "cosine": round(cs, 3),
                }
            )
    return {"within_slot": within, "slot_names": names, "across_slots": across}


def run_premise(p_idx, premise, bits, args, clients, embed, reused) -> dict:
    hosted = clients["openrouter"]
    if reused is not None:
        if reused["premise"] != premise:
            raise ValueError(f"premise {p_idx} differs from the reused run")
        raw = reused.get("raw_g_output")
        parsed = reused.get("slots") or []
    else:
        raw = llm(
            hosted,
            args.generator_model,
            SLOT_GENERATION_PROMPT.format(n=args.frame + args.slot_margin, premise=premise),
            temperature=0,
            top_p=1.0,
            max_tokens=4000,
            extra_body=LLAMACPP_NO_THINKING,
        )
        parsed = _parse_slots(raw)
    rec = {
        "prompt_idx": p_idx,
        "premise": premise,
        "n_requested": args.frame + args.slot_margin,
        "n_returned": reused["n_returned"] if reused is not None else len(parsed),
        "message_bits": bits,
    }
    if len(parsed) < args.frame:
        rec.update(ok=False, failure="too_few_slots", raw_g_output=raw)
        return rec
    slots = parsed[: args.frame]
    rec["slots"] = slots
    rec["duplicates"] = near_duplicates(slots, embed, args.jaccard, args.cosine)

    system = make_story(
        hosted,
        hosted,
        n_slots=args.frame,
        slot_margin=args.slot_margin,
        generator_model=args.generator_model,
        synth_client=clients["synth"],
        synth_model=args.synth_model,
        synth_extra_body=synth_body(args),
        **synth_sampling(args),
        decoder_client=hosted,
        decoder_model=args.decoder_model,
    )
    chunks = [[b] for b in bits]
    assigned = system._assign_bits_to_slots(slots, chunks)
    story = system._generate_story(premise, assigned)
    system.premise = premise
    system._error_encoded_length = args.frame
    recovered = system.recover_message(story, slots=slots)
    wrong = [i for i, (a, b) in enumerate(zip(bits, recovered)) if a != b]
    rec.update(
        ok=True,
        story=story,
        word_count=len(story.split()),
        recovered_bits=recovered,
        bit_errors=len(wrong),
        wrong_bit_positions=wrong,
        # Slot index each wrong bit lives in, via the key permutation.
        wrong_slots=[system._key_permutation(len(slots))[i] for i in wrong],
    )
    return rec


def synth_body(args) -> dict | None:
    if not args.synth_reasoning:
        return no_reasoning_body(args.synth_provider, args.synth_model)
    if args.synth_provider == "openrouter":
        return {"reasoning": {"enabled": True}}
    if args.synth_reasoning_effort:
        return {"reasoning_effort": args.synth_reasoning_effort}
    return None  # OpenAI: the model's default reasoning effort


def synth_sampling(args) -> dict:
    """make_story's writer sampling; empty keeps its defaults."""
    if args.synth_reasoning and args.synth_provider == "openai":
        return {"synth_temperature": None, "synth_top_p": None}
    return {}


def summarize(records: list[dict], frame: int) -> dict:
    ok = [r for r in records if r["ok"]]
    words = [r["word_count"] for r in ok]
    return {
        "frame": frame,
        "n_premises": len(records),
        "n_enough_slots": len(ok),
        "n_returned": sorted(r["n_returned"] for r in records),
        "within_slot_dups": sum(len(r["duplicates"]["within_slot"]) for r in ok),
        "slot_name_dups": sum(len(r["duplicates"]["slot_names"]) for r in ok),
        "across_slot_dup_pairs": sum(len(r["duplicates"]["across_slots"]) for r in ok),
        "premises_with_any_dup": sum(
            any(r["duplicates"][k] for k in r["duplicates"]) for r in ok
        ),
        "words": {
            "min": min(words, default=None),
            "median": float(np.median(words)) if words else None,
            "max": max(words, default=None),
        },
        "perfect_recovery": sum(r["bit_errors"] == 0 for r in ok),
        "bitwise_accuracy": (
            1 - sum(r["bit_errors"] for r in ok) / (frame * len(ok)) if ok else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame", type=int, default=32, help="Frame size F (bits = slots).")
    ap.add_argument("--slot-margin", type=int, default=2)
    ap.add_argument("--n-premises", type=int, default=20)
    ap.add_argument("--generator-model", default="qwen/qwen3.5-9b")
    ap.add_argument("--synth-model", default="deepseek/deepseek-v4.1-flash")
    ap.add_argument("--decoder-model", default="deepseek/deepseek-v4.1-flash")
    ap.add_argument("--embedding-model", default="text-embedding-3-large")
    ap.add_argument("--jaccard", type=float, default=0.6, help="Near-duplicate token overlap.")
    ap.add_argument("--cosine", type=float, default=0.85, help="Near-duplicate cosine.")
    ap.add_argument("--synth-provider", default="openrouter", choices=["openrouter", "openai"])
    ap.add_argument("--synth-reasoning", action="store_true",
                    help="Let the synthesizer reason (default: reasoning off).")
    ap.add_argument("--synth-reasoning-effort", default=None,
                    help="OpenAI reasoning_effort with --synth-reasoning "
                         "(default: the model's own).")
    ap.add_argument("--reuse-slots", type=Path, default=None,
                    help="pilot_records.jsonl of an earlier run at the same F: "
                         "use its G slots instead of calling G.")
    ap.add_argument("--tag", default="", help="Suffix for the output directory.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--data-dir", type=Path, default=Path("data/experiments"))
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = args.data_dir / "phase1_texts" / f"pilot_sg_f{args.frame}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = json.loads((args.data_dir / "prompts" / "story_prompts.json").read_text())
    premises = [p["premise"] for p in prompts["prompts"][: args.n_premises]]
    # Same messages the grid run at this F will use (phase1_generate --capacity).
    rng = np.random.default_rng(42 + args.frame)
    messages = rng.integers(0, 2, size=(300, args.frame)).tolist()

    reused = [None] * len(premises)
    if args.reuse_slots:
        by_idx = {}
        for line in args.reuse_slots.read_text().splitlines():
            r = json.loads(line)
            by_idx[r["prompt_idx"]] = r
        reused = [by_idx[i] for i in range(len(premises))]

    clients = {"openrouter": make_openrouter_client(), "synth": make_client(args.synth_provider)}
    oai = openai.OpenAI()

    def embed(texts: list[str]) -> np.ndarray:
        r = oai.embeddings.create(model=args.embedding_model, input=texts)
        v = np.array([d.embedding for d in r.data])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    records = []
    # StorySystem prints every slot it decodes; keep that out of the terminal.
    with open(out_dir / "decode_trace.log", "w") as trace, contextlib.redirect_stdout(trace):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_premise, i, p, messages[i], args, clients, embed, reused[i]
                ): i
                for i, p in enumerate(premises)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # keep the other premises going
                    rec = {"prompt_idx": i, "premise": premises[i], "ok": False,
                           "failure": repr(e), "n_returned": 0}
                records.append(rec)
                log.info(
                    "premise %d: returned %s slots, %s",
                    i,
                    rec.get("n_returned"),
                    f"{rec['bit_errors']} bit errors, {rec['word_count']} words"
                    if rec["ok"]
                    else rec["failure"],
                )

    records.sort(key=lambda r: r["prompt_idx"])
    with open(out_dir / "pilot_records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = summarize(records, args.frame)
    summary.update(
        {
            "generator_model": args.generator_model,
            "synth_model": args.synth_model,
            "decoder_model": args.decoder_model,
            "slot_margin": args.slot_margin,
            "synth_provider": args.synth_provider,
            "synth_extra_body": synth_body(args),
            "synth_sampling": synth_sampling(args) or "make_story defaults",
            "reused_slots_from": str(args.reuse_slots) if args.reuse_slots else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
