"""Expand Phase 1 prompt pools and regenerate messages.json.

Brings the prompt/seed artifacts under `data/experiments/prompts/` to the sizes
required by the Phase 1 plan in `experiment.md`:

- TopicQA: 50 -> 300 questions (GPT-4.1 one-shot, style-anchored on existing 50)
- Story:   20 -> 300 premises  (GPT-4.1 one-shot, style-anchored on existing 20)
- LitReview: 50 -> 300 paper indices (seeded permutation over the 1200-paper pool,
  excluding the 50 already picked)
- messages.json: regenerated from scratch with 300 stego + 300 c1 bit strings
  per system using a single numpy.default_rng(42) sequence.

Existing prompt IDs are preserved; new entries are appended with contiguous IDs
(50..299 / 20..299). Each regenerated file records provenance ("seed", "expansion").

Usage:
    python -m experiments.utils.expand_prompts --target topicqa
    python -m experiments.utils.expand_prompts --target story
    python -m experiments.utils.expand_prompts --target litreview
    python -m experiments.utils.expand_prompts --target messages
    python -m experiments.utils.expand_prompts --target all
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.utils.system_factory import make_clients
from systems.paths import litreview_papers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEED = 42
PROMPTS_DIR = Path("data/experiments/prompts")
PAPERS_PATH = litreview_papers()

TARGET_TOPICQA = 300
TARGET_STORY = 300
TARGET_LITREVIEW = 300
TARGET_BASELINE = 300
TARGET_MESSAGES = 300

GPT_MODEL = "gpt-4.1"
GPT_TEMPERATURE = 0.9


# ---------------------------------------------------------------------------
# GPT helper
# ---------------------------------------------------------------------------


def _gpt_json_call(
    client, system_prompt: str, user_prompt: str, max_tokens: int = 16000
) -> dict:
    """Call GPT-4.1 with JSON mode and retry. Returns parsed JSON dict."""
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=GPT_MODEL,
                temperature=GPT_TEMPERATURE,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                raise
            log.warning(f"GPT call retry {attempt + 1}: {e}")
            time.sleep(2**attempt)

    raise TimeoutError()


# ---------------------------------------------------------------------------
# TopicQA
# ---------------------------------------------------------------------------


def expand_topicqa(client) -> None:
    path = PROMPTS_DIR / "topicqa_prompts.json"
    data = json.loads(path.read_text())
    existing = data["prompts"]
    if len(existing) >= TARGET_TOPICQA:
        log.info(f"topicqa already at {len(existing)}, skipping")
        return

    need = TARGET_TOPICQA - len(existing)
    log.info(f"topicqa: {len(existing)} -> {TARGET_TOPICQA} (need {need} new)")

    existing_questions = [p["question"] for p in existing]
    seen = {q.strip().lower() for q in existing_questions}
    new_questions: list[str] = []
    rounds = 0

    system_prompt = (
        "You generate diverse, high-quality general-interest questions for a writing-task "
        "corpus. Each question must be open-ended enough to answer in ~300 words, span broad "
        "domains (policy, science, technology, society, culture, health, economics, ethics, "
        "education, environment, history, psychology, international affairs), and avoid "
        "duplicating any existing example in phrasing or topic."
    )

    while len(new_questions) < need and rounds < 5:
        remaining = need - len(new_questions)
        batch = min(remaining + 20, 260)  # overshoot to absorb dedup loss
        examples_block = "\n".join(f"- {q}" for q in existing_questions[:50])
        user_prompt = (
            f"Here are {len(existing_questions[:50])} existing questions used as style anchors:\n\n"
            f"{examples_block}\n\n"
            f"Generate {batch} NEW questions in the same style. They must:\n"
            f"- Be general-interest and answerable in roughly 300 words.\n"
            f"- Cover a wide range of domains and avoid clustering on any single topic.\n"
            f"- Not duplicate or closely paraphrase any of the examples above or each other.\n"
            f"- Be a single sentence ending in a question mark.\n\n"
            f'Return JSON of the form: {{"questions": ["...", "..."]}}'
        )
        log.info(f"topicqa GPT call round {rounds + 1}: asking for {batch}")
        resp = _gpt_json_call(client, system_prompt, user_prompt)
        got = resp.get("questions", [])
        log.info(f"  received {len(got)}; deduping")
        for q in got:
            if not isinstance(q, str):
                continue
            q = q.strip()
            key = q.lower()
            if not q or key in seen:
                continue
            seen.add(key)
            new_questions.append(q)
            if len(new_questions) >= need:
                break
        rounds += 1

    if len(new_questions) < need:
        raise RuntimeError(
            f"topicqa expansion fell short: got {len(new_questions)}/{need} after {rounds} rounds"
        )

    next_id = len(existing)
    for q in new_questions[:need]:
        existing.append({"id": next_id, "question": q})
        next_id += 1

    data["prompts"] = existing
    data["expansion"] = {
        "method": f"{GPT_MODEL} one-shot",
        "temperature": GPT_TEMPERATURE,
        "new_ids": [50, TARGET_TOPICQA - 1],
        "rounds": rounds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info(f"topicqa: wrote {len(existing)} prompts to {path}")


# ---------------------------------------------------------------------------
# Story
# ---------------------------------------------------------------------------


def expand_story(client) -> None:
    path = PROMPTS_DIR / "story_prompts.json"
    data = json.loads(path.read_text())
    existing = data["prompts"]
    if len(existing) >= TARGET_STORY:
        log.info(f"story already at {len(existing)}, skipping")
        return

    need = TARGET_STORY - len(existing)
    log.info(f"story: {len(existing)} -> {TARGET_STORY} (need {need} new)")

    existing_premises = [p["premise"] for p in existing]
    seen = {p.strip().lower() for p in existing_premises}
    new_premises: list[str] = []
    rounds = 0

    system_prompt = (
        "You generate diverse story premises for a fiction-writing corpus. Each premise is "
        "2-3 sentences, introduces a concrete protagonist in a specific setting, and ends on "
        "a hook that promises conflict or mystery. Premises should span genres (literary, "
        "speculative, thriller, mystery, historical, near-future sci-fi, magical realism) "
        "and avoid repeating character archetypes or settings across examples."
    )

    while len(new_premises) < need and rounds < 5:
        remaining = need - len(new_premises)
        batch = min(remaining + 20, 150)
        examples_block = "\n".join(f"- {p}" for p in existing_premises[:20])
        user_prompt = (
            f"Here are {len(existing_premises[:20])} existing premises as style anchors:\n\n"
            f"{examples_block}\n\n"
            f"Generate {batch} NEW story premises in the same style. They must:\n"
            f"- Be 2-3 sentences long, introducing a concrete protagonist and setting.\n"
            f"- End on a hook (mystery, conflict, decision, revelation).\n"
            f"- Span multiple genres; avoid reusing character types (detective, biologist, etc.) "
            f"across premises or with the examples.\n"
            f"- Not duplicate or closely paraphrase any of the examples above or each other.\n\n"
            f'Return JSON of the form: {{"premises": ["...", "..."]}}'
        )
        log.info(f"story GPT call round {rounds + 1}: asking for {batch}")
        resp = _gpt_json_call(client, system_prompt, user_prompt)
        got = resp.get("premises", [])
        log.info(f"  received {len(got)}; deduping")
        for p in got:
            if not isinstance(p, str):
                continue
            p = p.strip()
            key = p.lower()
            if not p or key in seen:
                continue
            seen.add(key)
            new_premises.append(p)
            if len(new_premises) >= need:
                break
        rounds += 1

    if len(new_premises) < need:
        raise RuntimeError(
            f"story expansion fell short: got {len(new_premises)}/{need} after {rounds} rounds"
        )

    next_id = len(existing)
    for p in new_premises[:need]:
        existing.append({"id": next_id, "premise": p})
        next_id += 1

    data["prompts"] = existing
    data["expansion"] = {
        "method": f"{GPT_MODEL} one-shot",
        "temperature": GPT_TEMPERATURE,
        "new_ids": [20, TARGET_STORY - 1],
        "rounds": rounds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info(f"story: wrote {len(existing)} prompts to {path}")


# ---------------------------------------------------------------------------
# LitReview
# ---------------------------------------------------------------------------


def expand_litreview() -> None:
    path = PROMPTS_DIR / "litreview_indices.json"
    data = json.loads(path.read_text())

    # Index space MUST match what phase1_generate uses: `system.corpus[corpus_idx]`
    # where `system.corpus = load_corpus(...)`. That list (1191 papers, in dict
    # iteration order) is NOT the same as papers.jsonl line order. Sourcing from
    # papers.jsonl previously caused index drift.
    from systems.core.litreview import load_corpus, prepare_references
    from systems.paths import litreview_references

    corpus = load_corpus(*litreview_references())
    num_papers = len(corpus)

    # Filter on USABLE refs (post-`prepare_references`), not raw `referenceCount`.
    # `prepare_references` drops refs lacking author/year/title, with <5-word titles,
    # or duplicate (author, year). A 20-bit greedy encoding needs comfortable
    # headroom over 40 refs to handle adversarial bit patterns.
    MIN_USABLE_REFS = 60
    eligible = [
        i
        for i, p in enumerate(corpus)
        if len(prepare_references(p["references"])) >= MIN_USABLE_REFS
    ]
    eligible_set = set(eligible)
    log.info(
        f"litreview corpus size (load_corpus): {num_papers}, "
        f"eligible (>= {MIN_USABLE_REFS} usable refs): {len(eligible)}"
    )

    # Existing indices were sampled in the wrong index space and cannot be
    # preserved — discard them and re-pick the full 300 deterministically.
    pre_filter = len(data.get("indices", []))
    if pre_filter > 0:
        log.warning(
            f"discarding {pre_filter} existing indices: previously sampled against "
            f"papers.jsonl (1200 entries) instead of load_corpus (1191 entries) — "
            f"index spaces do not align"
        )

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(num_papers).tolist()
    new_pool = [int(i) for i in perm if int(i) in eligible_set]
    if len(new_pool) < TARGET_LITREVIEW:
        raise RuntimeError(
            f"not enough eligible indices: need {TARGET_LITREVIEW}, pool has {len(new_pool)}"
        )
    new_indices = new_pool[:TARGET_LITREVIEW]

    data["seed"] = SEED
    data["indices"] = new_indices
    data["expansion"] = {
        "method": (
            f"numpy.default_rng({SEED}).permutation over load_corpus indices, "
            f"filtered to usable_refs>={MIN_USABLE_REFS} (post-prepare_references)"
        ),
        "seed": SEED,
        "min_usable_refs": MIN_USABLE_REFS,
        "corpus_source": "load_corpus(corpus.jsonl, references.jsonl)",
        "pool_size": num_papers,
        "eligible_pool_size": len(eligible),
        "new_count": TARGET_LITREVIEW,
        "new_ids_range": [0, TARGET_LITREVIEW - 1],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2))
    log.info(f"litreview: wrote {len(data['indices'])} indices to {path}")


# ---------------------------------------------------------------------------
# Baseline (corporate-email opening seeds for SentenceStegSystem)
# ---------------------------------------------------------------------------


BASELINE_SEED_EXAMPLES = [
    "Following up on yesterday's sync, I wanted to consolidate the action items before our Thursday review.",
    "Could you confirm whether the Q3 vendor onboarding deck has been routed through Legal for the standard clause review?",
    "Quick heads-up: the staging migration window is shifting from Tuesday 02:00 UTC to Wednesday 04:00 UTC.",
    "I've attached the latest variance report; please flag any cost-center anomalies before Friday close.",
    "Per our conversation, I'd like to schedule a 30-minute walkthrough of the revised escalation matrix.",
    "Reminder that the AP reconciliation packet is due by EOD tomorrow with the updated FX rates applied.",
    "Wanted to share initial thoughts on the proposed reorg of the platform reliability function.",
    "FYI — the procurement portal will be in read-only mode this weekend for the SSO migration.",
    "Tagging you in case Compliance flagged anything during the latest control-attestation walkthrough.",
    "Just got off a call with the regional ops lead; there are a few open items I'd like your read on.",
]


def expand_baseline(client) -> None:
    path = PROMPTS_DIR / "baseline_prompts.json"
    if path.exists():
        data = json.loads(path.read_text())
        existing = data.get("prompts", [])
    else:
        data = {}
        existing = []

    if len(existing) >= TARGET_BASELINE:
        log.info(f"baseline already at {len(existing)}, skipping")
        return

    need = TARGET_BASELINE - len(existing)
    log.info(f"baseline: {len(existing)} -> {TARGET_BASELINE} (need {need} new)")

    existing_seeds = [p["seed"] for p in existing]
    seen = {s.strip().lower() for s in existing_seeds}
    new_seeds: list[str] = []
    rounds = 0

    system_prompt = (
        "You generate diverse opening lines for internal corporate emails. Each opening is a "
        "single sentence (or two short sentences), reads as the natural first line of a real "
        "workplace email, and spans common business topics: project updates, meetings, process "
        "improvements, risk/compliance, HR, technical system requirements, business strategy, "
        "vendor management, finance. Avoid greetings ('Hi team,'), salutations, or sign-offs — "
        "start mid-thought as professionals do."
    )

    anchors = existing_seeds[:20] if existing_seeds else BASELINE_SEED_EXAMPLES

    while len(new_seeds) < need and rounds < 8:
        remaining = need - len(new_seeds)
        batch = min(remaining + 30, 120)
        examples_block = "\n".join(f"- {s}" for s in anchors)
        user_prompt = (
            f"Here are {len(anchors)} style anchors for corporate email openings:\n\n"
            f"{examples_block}\n\n"
            f"Generate {batch} NEW corporate email opening lines in the same style. They must:\n"
            f"- Be 1–2 sentences, ~10–40 words.\n"
            f"- Read as the natural opening line of a workplace email (no greeting, no signoff).\n"
            f"- Span project updates, meetings, process, compliance, HR, technical systems, "
            f"strategy, vendor management, finance. Avoid clustering on any single topic.\n"
            f"- Use professional yet natural tone; light use of acronyms is fine.\n"
            f"- Not duplicate or closely paraphrase the anchors or each other.\n\n"
            f'Return JSON of the form: {{"openings": ["...", "..."]}}'
        )
        log.info(f"baseline GPT call round {rounds + 1}: asking for {batch}")
        resp = _gpt_json_call(client, system_prompt, user_prompt)
        got = resp.get("openings", [])
        log.info(f"  received {len(got)}; deduping")
        for s in got:
            if not isinstance(s, str):
                continue
            s = s.strip()
            key = s.lower()
            if not s or key in seen:
                continue
            seen.add(key)
            new_seeds.append(s)
            if len(new_seeds) >= need:
                break
        rounds += 1

    if len(new_seeds) < need:
        raise RuntimeError(
            f"baseline expansion fell short: got {len(new_seeds)}/{need} after {rounds} rounds"
        )

    next_id = len(existing)
    for s in new_seeds[:need]:
        existing.append({"id": next_id, "seed": s})
        next_id += 1

    data["prompts"] = existing
    data["expansion"] = {
        "method": f"{GPT_MODEL} one-shot",
        "temperature": GPT_TEMPERATURE,
        "new_ids": [0, TARGET_BASELINE - 1],
        "rounds": rounds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info(f"baseline: wrote {len(existing)} prompts to {path}")


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


MESSAGE_SPEC = [
    ("topicqa", 6),
    ("story", 18),
    ("litreview", 20),
    ("baseline", 3),
]


def regenerate_messages() -> None:
    path = PROMPTS_DIR / "messages.json"
    rng = np.random.default_rng(SEED)

    out: dict = {
        "seed": SEED,
        "rng": "numpy.default_rng",
        "draw_order": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for system, num_bits in MESSAGE_SPEC:
        stego = rng.integers(0, 2, size=(TARGET_MESSAGES, num_bits)).tolist()
        c1 = rng.integers(0, 2, size=(TARGET_MESSAGES, num_bits)).tolist()
        out[system] = {
            "num_bits": num_bits,
            "stego_messages": stego,
            "c1_messages": c1,
        }
        out["draw_order"].append(f"{system}.stego")
        out["draw_order"].append(f"{system}.c1")

    path.write_text(json.dumps(out, indent=2))
    log.info(f"messages.json: wrote 300 stego + 300 c1 per system to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["topicqa", "story", "litreview", "baseline", "messages", "all"],
        default="all",
    )
    args = parser.parse_args()

    client = None
    if args.target in ("topicqa", "story", "baseline", "all"):
        client, _ = make_clients()

    if args.target in ("topicqa", "all"):
        expand_topicqa(client)
    if args.target in ("story", "all"):
        expand_story(client)
    if args.target in ("litreview", "all"):
        expand_litreview()
    if args.target in ("baseline", "all"):
        expand_baseline(client)
    if args.target in ("messages", "all"):
        regenerate_messages()

    log.info("Expansion complete.")


if __name__ == "__main__":
    main()
