"""Exp 1: Subtopic generation determinism (local model).
Run generate_subtopics 10x per question via local greedy decode.
Every run should be bitwise identical. Goal: 100% identity.
"""
from common import QUESTIONS, generate_subtopics, canonicalize, N_SUBTOPICS
import json

RUNS = 5

results = []
for qi, q in enumerate(QUESTIONS):
    print(f"\n[{qi+1}/{len(QUESTIONS)}] {q[:60]}...")
    runs = []
    for r in range(RUNS):
        topics = generate_subtopics(q, N_SUBTOPICS)
        runs.append(topics)
        if r == 0:
            print(f"  {len(topics)} topics: {topics[:3]}...")

    # Check all runs identical
    identical = all(r == runs[0] for r in runs[1:])
    n_unique = len(set(json.dumps(r) for r in runs))

    results.append({
        "question": q,
        "identical": identical,
        "n_unique_lists": n_unique,
        "n_topics": len(runs[0]),
        "sample": runs[0][:5],
    })
    status = "PASS" if identical else f"FAIL ({n_unique} distinct lists)"
    print(f"  {status}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
n_pass = sum(1 for r in results if r["identical"])
print(f"Deterministic: {n_pass}/{len(results)} questions")
print(f"Goal: {len(results)}/{len(results)}")

with open("exp1_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved exp1_results.json")
