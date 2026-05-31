import asyncio
import json
import os
import re
import time
from pathlib import Path

import aiohttp
import numpy as np
from tqdm import tqdm

#### Configuration

API_BASE = "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"

NUM_SEED_QUESTIONS = 80
TRACES_PER_QUESTION = 40
STEPS_PER_TRACE = 5
PARAPHRASES_PER_STEP = 2

# Concurrency limits
TRACE_CONCURRENCY = 30  # traces in parallel (steps within a trace are serial)
PARAPHRASE_CONCURRENCY = 50
EMBED_CONCURRENCY = 5
CHECKPOINT_INTERVAL = 100

OUT_DIR = Path("./pca/cot_reasoning/artifacts/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS_JSON = OUT_DIR / "seed_questions.json"
TRACES_JSON = OUT_DIR / "reasoning_traces.json"
STEPS_JSON = OUT_DIR / "sentences.json"
PARAPHRASE_JSON = OUT_DIR / "paraphrase_pairs.json"
EMBED_NPY = OUT_DIR / "sentence_embeddings.npy"
PARAPHRASE_EMBED_NPY = OUT_DIR / "paraphrase_embeddings.npy"
GROUP_LABELS_NPY = OUT_DIR / "group_labels.npy"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

### Prompts

QUESTION_GEN_SYSTEM = """You are generating causal reasoning questions for a dataset. Each question should:
1. Require a multi-step causal chain to answer (at least 4-5 distinct causal links)
2. Be about concrete, real-world mechanisms (not abstract philosophy)
3. Start with "Why" or "How does" or "What causes"
4. Be answerable through a chain of cause-effect reasoning where each step depends on the previous

Generate questions from DIVERSE domains. Output one question per line, no numbering."""

QUESTION_GEN_DOMAINS = [
    "climate and weather systems",
    "supply chains and economics",
    "biological processes and ecology",
    "engineering failures and material science",
    "public health and epidemiology",
    "urban planning and infrastructure",
    "agricultural systems and food production",
    "energy systems and power grids",
    "ocean and marine systems",
    "transportation and logistics",
    "communication networks and information flow",
    "geological processes and natural disasters",
    "industrial chemistry and manufacturing",
    "water systems and hydrology",
    "social dynamics and institutional behavior",
    "aerospace and aviation systems",
]

STEP_SYSTEM = """You are reasoning through a causal question step by step. Each step must:
1. Identify ONE specific causal factor or mechanism
2. Explain HOW it connects to the previous step's conclusion
3. Name the specific entity, process, or mechanism involved (be concrete)
4. Be 20-40 words, self-contained, written as a declarative claim

You are writing ONLY the next step. Be specific and concrete — name mechanisms, substances, processes, or entities. Vary your framing: sometimes lead with the cause, sometimes with the effect, sometimes with the mechanism."""

STEP_TEMPLATE = """Question: {question}

Previous reasoning steps:
{previous_steps}

Write step {step_num}: identify the next causal factor and explain how it follows from the previous step."""

STEP_TEMPLATE_FIRST = """Question: {question}

Write step 1: identify the first relevant causal factor that begins to answer this question."""

PARAPHRASE_SYSTEM = """You are a text paraphrasing assistant. Rewrite the given reasoning step while:
1. PRESERVING the causal claim and all specific entities/mechanisms mentioned
2. CHANGING sentence structure significantly
3. USING different vocabulary where possible
4. MAINTAINING the same logical content

Output ONLY the paraphrased text. No commentary."""


### Async API helpers


async def chat_completion(
    session: aiohttp.ClientSession,
    messages: list[dict],
    temperature: float = 0.8,
    max_retries: int = 5,
) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 300,
    }
    for attempt in range(max_retries):
        try:
            async with session.post(
                f"{API_BASE}/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as r:
                if r.status == 429:
                    retry_after = float(r.headers.get("Retry-After", 2**attempt))
                    await asyncio.sleep(retry_after)
                    continue
                r.raise_for_status()
                data = await r.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return ""
            await asyncio.sleep(2**attempt)
    return ""


async def embed_batch(
    session: aiohttp.ClientSession,
    texts: list[str],
    max_retries: int = 5,
) -> list[list[float]]:
    payload = {"model": EMBED_MODEL, "input": texts}
    for attempt in range(max_retries):
        try:
            async with session.post(
                f"{API_BASE}/embeddings",
                headers=HEADERS,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as r:
                if r.status == 429:
                    retry_after = float(r.headers.get("Retry-After", 2**attempt))
                    await asyncio.sleep(retry_after)
                    continue
                r.raise_for_status()
                data = await r.json()
                return [d["embedding"] for d in data["data"]]
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2**attempt)
    return []


### IO helpers


def load_or_init_json(path: Path) -> list:
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {path.name}")
        return data
    return []


def save_json(data: list, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


### Phase 0: Seed questions


async def generate_seed_questions(session: aiohttp.ClientSession) -> list[dict]:
    questions = load_or_init_json(QUESTIONS_JSON)
    if len(questions) >= NUM_SEED_QUESTIONS:
        return questions[:NUM_SEED_QUESTIONS]

    existing_texts = {q["question"] for q in questions}
    per_domain = max(
        1, (NUM_SEED_QUESTIONS - len(questions)) // len(QUESTION_GEN_DOMAINS) + 1
    )

    for domain in QUESTION_GEN_DOMAINS:
        if len(questions) >= NUM_SEED_QUESTIONS:
            break
        messages = [
            {"role": "system", "content": QUESTION_GEN_SYSTEM},
            {
                "role": "user",
                "content": f"Generate {per_domain} causal reasoning questions about: {domain}",
            },
        ]
        response = await chat_completion(session, messages, temperature=0.9)
        for line in response.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if not line or line in existing_texts:
                continue
            questions.append(
                {
                    "question_id": len(questions),
                    "domain": domain,
                    "question": line,
                }
            )
            existing_texts.add(line)
            if len(questions) >= NUM_SEED_QUESTIONS:
                break

    save_json(questions, QUESTIONS_JSON)
    print(f"Generated {len(questions)} seed questions")
    return questions[:NUM_SEED_QUESTIONS]


### Phase 1: Reasoning traces


async def generate_single_trace(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    qid: int,
    question_text: str,
    trace_id: int,
) -> dict | None:
    async with semaphore:
        steps = []
        for step_num in range(1, STEPS_PER_TRACE + 1):
            if step_num == 1:
                prompt = STEP_TEMPLATE_FIRST.format(question=question_text)
            else:
                prev = "\n".join(f"  Step {i + 1}: {s}" for i, s in enumerate(steps))
                prompt = STEP_TEMPLATE.format(
                    question=question_text,
                    previous_steps=prev,
                    step_num=step_num,
                )
            messages = [
                {"role": "system", "content": STEP_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            step_text = await chat_completion(session, messages, temperature=1.3)
            step_text = step_text.strip().strip('"')
            step_text = re.sub(
                r"^(?:Step\s+\d+\s*[:.\-]\s*)", "", step_text, flags=re.IGNORECASE
            ).strip()
            if not step_text:
                return None
            steps.append(step_text)
        return {
            "question_id": qid,
            "trace_id": trace_id,
            "question": question_text,
            "steps": steps,
        }


async def generate_reasoning_traces(
    session: aiohttp.ClientSession,
    questions: list[dict],
) -> list[dict]:
    traces = load_or_init_json(TRACES_JSON)
    existing_keys = {(t["question_id"], t["trace_id"]) for t in traces}

    todo = []
    for q in questions:
        qid = q["question_id"]
        for trace_id in range(TRACES_PER_QUESTION):
            if (qid, trace_id) not in existing_keys:
                todo.append((qid, q["question"], trace_id))

    if not todo:
        print(f"All {len(traces)} traces already generated")
        return traces

    print(f"Generating {len(todo)} traces ({TRACE_CONCURRENCY} concurrent)...")
    semaphore = asyncio.Semaphore(TRACE_CONCURRENCY)
    lock = asyncio.Lock()
    pbar = tqdm(total=len(todo), desc="Traces")
    completed = 0

    async def run_and_collect(qid, question_text, trace_id):
        nonlocal completed
        result = await generate_single_trace(
            session, semaphore, qid, question_text, trace_id
        )
        async with lock:
            if result is not None:
                traces.append(result)
            completed += 1
            pbar.update(1)
            if completed % CHECKPOINT_INTERVAL == 0:
                save_json(traces, TRACES_JSON)

    await asyncio.gather(*(run_and_collect(q, qt, t) for q, qt, t in todo))
    pbar.close()
    save_json(traces, TRACES_JSON)
    return traces


### Phase 2: Flatten


def flatten_to_sentences(traces: list[dict]) -> list[dict]:
    sentences = []
    for trace in traces:
        qid = trace["question_id"]
        for step_pos, step_text in enumerate(trace["steps"]):
            flat_idx = qid * STEPS_PER_TRACE + step_pos
            sentences.append(
                {
                    "flat_idx": flat_idx,
                    "question_id": qid,
                    "trace_id": trace["trace_id"],
                    "step_position": step_pos,
                    "text": step_text,
                }
            )
    save_json(sentences, STEPS_JSON)
    return sentences


### Phase 3: Paraphrases


async def generate_paraphrases(
    session: aiohttp.ClientSession,
    sentences: list[dict],
) -> list[dict]:
    if PARAPHRASES_PER_STEP == 0:
        return []

    paraphrases = load_or_init_json(PARAPHRASE_JSON)
    existing_keys = {
        (p["question_id"], p["trace_id"], p["step_position"], p["paraphrase_index"])
        for p in paraphrases
    }

    todo = []
    for sent in sentences:
        for para_idx in range(PARAPHRASES_PER_STEP):
            key = (
                sent["question_id"],
                sent["trace_id"],
                sent["step_position"],
                para_idx,
            )
            if key not in existing_keys:
                todo.append((sent, para_idx))

    if not todo:
        print(f"All {len(paraphrases)} paraphrases already generated")
        return paraphrases

    print(
        f"Generating {len(todo)} paraphrases ({PARAPHRASE_CONCURRENCY} concurrent)..."
    )
    semaphore = asyncio.Semaphore(PARAPHRASE_CONCURRENCY)
    lock = asyncio.Lock()
    pbar = tqdm(total=len(todo), desc="Paraphrases")
    completed = 0

    async def run_one(sent, para_idx):
        nonlocal completed
        async with semaphore:
            messages = [
                {"role": "system", "content": PARAPHRASE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Paraphrase this reasoning step:\n\n{sent['text']}",
                },
            ]
            paraphrased = await chat_completion(session, messages, temperature=0.7)
            paraphrased = paraphrased.strip().strip('"')
        async with lock:
            if paraphrased:
                paraphrases.append(
                    {
                        "question_id": sent["question_id"],
                        "trace_id": sent["trace_id"],
                        "step_position": sent["step_position"],
                        "paraphrase_index": para_idx,
                        "original_text": sent["text"],
                        "paraphrased_text": paraphrased,
                    }
                )
            completed += 1
            pbar.update(1)
            if completed % (CHECKPOINT_INTERVAL * 2) == 0:
                save_json(paraphrases, PARAPHRASE_JSON)

    await asyncio.gather(*(run_one(s, pi) for s, pi in todo))
    pbar.close()
    save_json(paraphrases, PARAPHRASE_JSON)
    return paraphrases


### Phase 4: Embeddings


async def compute_embeddings(
    session: aiohttp.ClientSession,
    sentences: list[dict],
    paraphrases: list[dict],
):
    batch_size = 512
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)

    async def embed_all(texts: list[str], out_path: Path) -> np.ndarray:
        if out_path.exists():
            existing = np.load(out_path)
            if len(existing) >= len(texts):
                return existing[: len(texts)]
            start = len(existing)
            print(f"Resuming embeddings from index {start}")
        else:
            existing = None
            start = 0

        remaining = texts[start:]
        n_batches = (len(remaining) + batch_size - 1) // batch_size
        results = [None] * n_batches

        async def do_batch(idx):
            async with semaphore:
                batch = remaining[idx * batch_size : (idx + 1) * batch_size]
                results[idx] = await embed_batch(session, batch)

        await asyncio.gather(*(do_batch(i) for i in range(n_batches)))

        new_embs = []
        for r in results:
            if r:
                new_embs.extend(r)
        new_arr = np.array(new_embs)

        if existing is not None:
            arr = np.vstack([existing, new_arr])
        else:
            arr = new_arr
        np.save(out_path, arr)
        return arr

    step_texts = [s["text"] for s in sentences]
    step_emb = await embed_all(step_texts, EMBED_NPY)
    print(f"Step embeddings: {step_emb.shape}")

    if paraphrases:
        para_texts = [p["paraphrased_text"] for p in paraphrases]
        para_emb = await embed_all(para_texts, PARAPHRASE_EMBED_NPY)
        print(f"Paraphrase embeddings: {para_emb.shape}")

        sent_key_to_idx = {}
        for i, s in enumerate(sentences):
            sent_key_to_idx[(s["question_id"], s["trace_id"], s["step_position"])] = i
        group_labels = np.array(
            [
                sent_key_to_idx[(p["question_id"], p["trace_id"], p["step_position"])]
                for p in paraphrases
            ]
        )
        np.save(GROUP_LABELS_NPY, group_labels)
        print(f"Group labels: {group_labels.shape}")


### Main


async def main():
    print("-" * 60)
    print("CoT Reasoning Dataset Generation (async)")
    print(
        f"  {NUM_SEED_QUESTIONS} questions × {TRACES_PER_QUESTION} traces × {STEPS_PER_TRACE} steps"
    )
    total_steps = NUM_SEED_QUESTIONS * TRACES_PER_QUESTION * STEPS_PER_TRACE
    print(f"  Expected steps: {total_steps}")
    print(f"  Expected paraphrases: {total_steps * PARAPHRASES_PER_STEP}")
    diff_per_group = TRACES_PER_QUESTION * (TRACES_PER_QUESTION - 1) // 2
    print(
        f"  Diff vectors: {NUM_SEED_QUESTIONS * STEPS_PER_TRACE} groups × {diff_per_group} = {NUM_SEED_QUESTIONS * STEPS_PER_TRACE * diff_per_group}"
    )
    print(
        f"  Concurrency: traces={TRACE_CONCURRENCY}, paraphrases={PARAPHRASE_CONCURRENCY}"
    )
    print("-" * 60)

    t0 = time.time()

    connector = aiohttp.TCPConnector(
        limit=max(TRACE_CONCURRENCY, PARAPHRASE_CONCURRENCY) + 10
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        print("\n[0/4] Generating seed questions...")
        questions = await generate_seed_questions(session)

        print("\n[1/4] Generating reasoning traces...")
        traces = await generate_reasoning_traces(session, questions)
        print(f"Total traces: {len(traces)}")

        print("\n[2/4] Flattening to sentences.json...")
        sentences = flatten_to_sentences(traces)
        print(f"Total steps: {len(sentences)}")
        n_groups = len({s["flat_idx"] for s in sentences})
        print(f"Unique groups (flat_idx): {n_groups}")

        print("\n[3/4] Generating paraphrases...")
        paraphrases = await generate_paraphrases(session, sentences)
        print(f"Total paraphrases: {len(paraphrases)}")

        print("\n[4/4] Computing embeddings...")
        await compute_embeddings(session, sentences, paraphrases)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed / 60:.1f} minutes")
    print(f"Artifacts saved to {OUT_DIR}")
    print("Ready for pca_train.py (update OUT_DIR to point here)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
