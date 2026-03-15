"""Exp 2: Encoding controllability.
Generate subtopics, select subsets, prompt model to cover exactly those.
LLM judge checks: (a) are selected subtopics covered? (b) are excluded subtopics absent?
Goal: >95% compliance.
"""

import json
import random

from common import (
    N_SUBTOPICS,
    QUESTIONS,
    encode_response,
    generate_subtopics,
    group_subtopics,
    llm,
    select_subtopics,
)

N_TRIALS = 20
KEY = "experiment2"

results = []
for ti in range(N_TRIALS):
    q = QUESTIONS[ti % len(QUESTIONS)]
    print(f"\n[{ti + 1}/{N_TRIALS}] {q[:60]}...")

    topics = generate_subtopics(q, N_SUBTOPICS)
    groups = group_subtopics(topics, KEY)
    bits = [random.randint(0, 1) for _ in groups]
    selected = select_subtopics(groups, bits)
    excluded = [t for t in topics if t not in selected]

    response = encode_response(q, selected)
    print(f"  selected: {selected[:3]}...")
    print(f"  response: {response[:120]}...")

    # LLM judge: check each topic
    sel_str = "\n".join(f"- {t}" for t in selected)
    exc_str = "\n".join(f"- {t}" for t in excluded)

    judge_prompt = f"""A response was generated for the question: "{q}"

Response:
---
{response}
---

INCLUDED topics (should be discussed):
{sel_str}

EXCLUDED topics (should NOT be discussed):
{exc_str}

For each included topic, say "COVERED" or "MISSING".
For each excluded topic, say "ABSENT" or "LEAKED".
Output as JSON: {{"included": {{"topic": "COVERED"|"MISSING", ...}}, "excluded": {{"topic": "ABSENT"|"LEAKED", ...}}}}"""

    raw = llm(judge_prompt, temperature=0)
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        verdict = json.loads(raw)
    except:
        print(f"  PARSE ERROR: {raw[:200]}")
        continue

    n_covered = sum(1 for v in verdict.get("included", {}).values() if v == "COVERED")
    n_selected = len(selected)
    n_absent = sum(1 for v in verdict.get("excluded", {}).values() if v == "ABSENT")
    n_excluded = len(excluded)

    coverage = n_covered / n_selected if n_selected else 1
    absence = n_absent / n_excluded if n_excluded else 1

    results.append(
        {
            "question": q,
            "coverage": coverage,
            "absence": absence,
            "n_covered": n_covered,
            "n_selected": n_selected,
            "n_absent": n_absent,
            "n_excluded": n_excluded,
        }
    )
    print(
        f"  coverage={coverage:.0%} ({n_covered}/{n_selected}) absence={absence:.0%} ({n_absent}/{n_excluded})"
    )

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
avg_cov = sum(r["coverage"] for r in results) / len(results)
avg_abs = sum(r["absence"] for r in results) / len(results)
print(f"Avg topic coverage:   {avg_cov:.1%}")
print(f"Avg exclusion rate:   {avg_abs:.1%}")
print("Goal: >95% coverage, >95% exclusion")

with open("exp2_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved exp2_results.json")
