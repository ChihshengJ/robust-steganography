"""Exp 3: Forced-choice decoder accuracy (no paraphrase).
Generate 50 responses with known subtopic selections. Run forced-choice decoder.
Test pairs (binary) and quads (4-way) separately.
Goal: >98% for pairs, >92% for quads.
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
    select_subtopics,
)

N_TRIALS = 50
KEY = "experiment3"


def run_trial(qi, group_size):
    q = QUESTIONS[qi % len(QUESTIONS)]
    n = (N_SUBTOPICS // group_size) * group_size  # round down to fit groups evenly
    topics = generate_subtopics(q, n)
    groups = group_subtopics(topics, KEY + str(qi), group_size=group_size)
    bits = [random.randint(0, group_size - 1) for _ in groups]
    selected = select_subtopics(groups, bits)
    response = encode_response(q, selected)

    decoded_bits = []
    for group in groups:
        decoded_bits.append(decode_group(response, q, group))

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
    print(f"Testing {label} (group_size={group_size})")
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
    print(f"\n{label} SUMMARY:")
    print(f"  Per-bit accuracy:     {overall:.1%}")
    print(f"  Perfect recovery:     {perfect:.1%}")

    with open(f"exp3_{label.lower()}_results.json", "w") as f:
        json.dump(
            {
                "overall_accuracy": overall,
                "perfect_recovery": perfect,
                "trials": results,
            },
            f,
            indent=2,
        )

print("\nGoals: pairs >98%, quads >92%")
