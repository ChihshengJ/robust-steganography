"""Exp 4: Forced-choice decoder accuracy (post paraphrase).
Same as exp3 but paraphrase responses before decoding.
Goal: >95% for pairs, >85% for quads.
"""

import json
import random

from common import (
    N_SUBTOPICS,
    QUESTIONS,
    decode_group,
    encode_response,
    generate_subtopics,
    group_subtopics,
    paraphrase,
    select_subtopics,
)

N_TRIALS = 50
KEY = "experiment4"


def run_trial(qi, group_size):
    q = QUESTIONS[qi % len(QUESTIONS)]
    n = (N_SUBTOPICS // group_size) * group_size
    topics = generate_subtopics(q, n)
    groups = group_subtopics(topics, KEY + str(qi), group_size=group_size)
    bits = [random.randint(0, group_size - 1) for _ in groups]
    selected = select_subtopics(groups, bits)

    response = encode_response(q, selected)
    paraphrased = paraphrase(response)

    decoded_bits = []
    for group in groups:
        decoded_bits.append(decode_group(paraphrased, q, group))

    correct = sum(1 for a, b in zip(bits, decoded_bits) if a == b)
    total = len(bits)
    return {
        "question": q,
        "bits": bits,
        "decoded": decoded_bits,
        "correct": correct,
        "total": total,
        "acc": correct / total,
    }


for group_size, label in [(2, "PAIRS"), (4, "QUADS")]:
    print(f"\n{'=' * 60}")
    print(f"Testing {label} (group_size={group_size}) — POST PARAPHRASE")
    print(f"{'=' * 60}")

    results = []
    total_correct = 0
    total_bits = 0

    for ti in range(N_TRIALS):
        print(f"  [{ti + 1}/{N_TRIALS}]", end=" ")
        r = run_trial(ti, group_size)
        results.append(r)
        total_correct += r["correct"]
        total_bits += r["total"]
        print(f"acc={r['acc']:.0%} ({r['correct']}/{r['total']})")

    overall = total_correct / total_bits
    perfect = sum(1 for r in results if r["correct"] == r["total"]) / len(results)
    print(f"\n{label} SUMMARY (post-paraphrase):")
    print(f"  Per-bit accuracy:     {overall:.1%}")
    print(f"  Perfect recovery:     {perfect:.1%}")

    with open(f"exp4_{label.lower()}_results.json", "w") as f:
        json.dump(
            {
                "overall_accuracy": overall,
                "perfect_recovery": perfect,
                "trials": results,
            },
            f,
            indent=2,
        )

print("\nGoals: pairs >95%, quads >85%")
