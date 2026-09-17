"""Exp 5: Subtopic leakage / cross-talk.
Check whether covering subtopic A from pair (A,B) causes decoder to detect B.
Focus on the 20 hardest pairs (most semantically similar subtopics).
Goal: cross-talk <10%.
"""

import json

from common import (
    N_SUBTOPICS,
    QUESTIONS,
    decode_group,
    encode_response,
    generate_subtopics,
    group_subtopics,
)

KEY = "experiment5"
N_QUESTIONS = 20


def similarity_score(a, b):
    """Cheap word-overlap proxy for semantic similarity."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0


results = []
for qi in range(N_QUESTIONS):
    q = QUESTIONS[qi]
    print(f"\n[{qi + 1}/{N_QUESTIONS}] {q[:60]}...")

    topics = generate_subtopics(q, N_SUBTOPICS)
    groups = group_subtopics(topics, KEY + str(qi), group_size=2)

    # Rank pairs by similarity, test all but flag hardest
    pair_scores = [(similarity_score(g[0], g[1]), gi, g) for gi, g in enumerate(groups)]
    pair_scores.sort(reverse=True)

    for sim, gi, group in pair_scores:
        for chosen_idx in [0, 1]:
            # Generate response covering chosen topic (and other selected topics from other groups)
            all_selected = []
            for gj, g2 in enumerate(groups):
                if gj == gi:
                    all_selected.append(g2[chosen_idx])
                else:
                    all_selected.append(g2[0])  # default to first

            response = encode_response(q, all_selected)

            # Decode just this group
            detected = decode_group(response, q, group)
            is_correct = detected == chosen_idx
            cross_talk = not is_correct

            results.append(
                {
                    "question": q,
                    "group": group,
                    "similarity": sim,
                    "chosen": chosen_idx,
                    "detected": detected,
                    "correct": is_correct,
                    "cross_talk": cross_talk,
                }
            )
            ct_label = "CROSSTALK" if cross_talk else "ok"
            print(
                f"  pair({group[0][:25]}, {group[1][:25]}) sim={sim:.2f} chose={chosen_idx} det={detected} {ct_label}"
            )

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
total = len(results)
ct_count = sum(1 for r in results if r["cross_talk"])
ct_rate = ct_count / total if total else 0
print(f"Total tests:     {total}")
print(f"Cross-talk:      {ct_count} ({ct_rate:.1%})")
print("Goal: <10% cross-talk")

# Break down by similarity quartile
sims = sorted(set(r["similarity"] for r in results))
if len(sims) >= 4:
    med = sims[len(sims) // 2]
    high_sim = [r for r in results if r["similarity"] >= med]
    low_sim = [r for r in results if r["similarity"] < med]
    hi_ct = (
        sum(1 for r in high_sim if r["cross_talk"]) / len(high_sim) if high_sim else 0
    )
    lo_ct = sum(1 for r in low_sim if r["cross_talk"]) / len(low_sim) if low_sim else 0
    print(f"High-similarity cross-talk: {hi_ct:.1%}")
    print(f"Low-similarity cross-talk:  {lo_ct:.1%}")

with open("exp5_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved exp5_results.json")
