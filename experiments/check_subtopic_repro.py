"""Check whether a llama.cpp serve command reproduces Phase 1's TopicQA subtopics.

TopicQA decodes by re-generating the 12 subtopics with the local model and
grouping them with the secret key. Decoding only works if that regeneration is
identical -- same phrases, SAME ORDER -- to what Phase 1 produced at encode
time (the key fixes the group permutation, not which phrase fills each slot, so
even a reordered list breaks decoding).

Phase 1 already saved the topics it generated, in
``data/experiments/phase1_texts/topicqa_stego.jsonl`` under ``metadata.topics``.
That saved list IS "the Phase 1 generation". This script only READS that file
(it never writes it), re-generates the subtopics for the same questions against
whatever local server you point it at, and reports how many reproduce Phase 1.

Workflow:
    1. Launch llama-server with the command you want to test.
    2. python -m experiments.check_subtopic_repro --base-url http://127.0.0.1:8080/v1
    3. Read the summary: "30/30 reproduce Phase 1 exactly" => that command works.

Local-server calls only -- no OpenAI / GPT cost.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openai

from experiments.utils.io import read_jsonl
from experiments.utils.system_factory import make_topicqa


def main():
    parser = argparse.ArgumentParser(
        description="Does a local serve command reproduce Phase 1's TopicQA subtopics?"
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080/v1",
        help="Local server to test (default: http://127.0.0.1:8080/v1).",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Model name to request (default: keep the factory value).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="How many Phase 1 questions to check (default 30).",
    )
    parser.add_argument(
        "--stego-file",
        type=Path,
        default=Path("data/experiments/phase1_texts/recovery_test/topicqa_stego.jsonl"),
        help="Phase 1 stego file to read the reference topics from (read-only).",
    )
    args = parser.parse_args()

    records = read_jsonl(args.stego_file)
    if not records:
        print(f"No records at {args.stego_file}")
        return
    records = sorted(records, key=lambda r: r.get("prompt_idx", 0))[: args.limit]
    print(f"Reference: {len(records)} questions from {args.stego_file} (read-only)")
    print(f"Testing server: {args.base_url}")
    print("Local-server calls only -- no OpenAI cost.\n")

    # Same TopicQA params as the real decode; the OpenAI client is never used
    # here (generate_subtopics is local-only) but the constructor needs one.
    client = openai.OpenAI(api_key="unused")
    local_client = openai.OpenAI(base_url=args.base_url, api_key="unused")
    system = make_topicqa(client, local_client)
    if args.local_model:
        system.local_model = args.local_model

    n_exact = n_reordered = n_drift = n_error = 0

    for rec in records:
        rid = rec["id"]
        stored = (rec.get("metadata") or {}).get("topics")
        question = (rec.get("system_state") or {}).get("question") or rec.get("prompt")
        if not stored or not question:
            print(f"[{rid}] skipped -- no stored topics/question")
            continue

        try:
            regen = system.generate_subtopics(question)
        except Exception as e:
            n_error += 1
            print(f"[{rid}] SERVER ERROR: {e!r}")
            continue

        if regen == stored:
            n_exact += 1
            print(f"[{rid}] MATCH")
        elif set(regen) == set(stored):
            n_reordered += 1
            print(
                f"[{rid}] REORDERED -- same 12 phrases, different order "
                f"(still breaks decoding)"
            )
        else:
            n_drift += 1
            shared = len(set(regen) & set(stored))
            print(f"[{rid}] DIFFERENT -- {shared}/{len(stored)} phrases shared")
            print(f"    Phase 1: {stored}")
            print(f"    regen:   {regen}")

    n_tested = n_exact + n_reordered + n_drift
    print("\n" + "=" * 60)
    if n_tested == 0:
        print(f"Nothing tested ({n_error} server errors). Is the server up?")
        return
    print(f"reproduce Phase 1 exactly .. {n_exact}/{n_tested}   <- only these decode")
    print(f"same phrases, reordered .... {n_reordered}/{n_tested}")
    print(f"content drift .............. {n_drift}/{n_tested}")
    if n_error:
        print(f"server errors .............. {n_error}")
    print("=" * 60)
    if n_exact == n_tested:
        print("VERDICT: this serve command reproduces Phase 1 -- decode with it.")
    elif n_exact == 0:
        print("VERDICT: this serve command does NOT reproduce Phase 1 at all.")
    else:
        print(
            "VERDICT: partial -- this serve command does not reliably reproduce "
            "Phase 1."
        )


if __name__ == "__main__":
    main()
