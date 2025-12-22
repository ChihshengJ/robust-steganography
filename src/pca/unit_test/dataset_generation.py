import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from datasets import Dataset, load_dataset
from tqdm import tqdm


API_BASE = "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"

N_TESTS_PER_PROBLEM = 15
PCA_COMPONENTS = 3
OUT_DIR = Path("./src/pca/unit_test/artifacts/")
OUT_DIR.mkdir(exist_ok=True)

TESTS_JSON = OUT_DIR / "generated_tests.json"
EMBED_NPY = OUT_DIR / "embeddings.npy"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def chat_completion(messages, temperature=0.8):
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(
        f"{API_BASE}/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def embed_texts_chunked(texts, batch_size=512):
    """
    Embed a list of texts by chunking into batches.
    Preserves input order exactly.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        payload = {
            "model": EMBED_MODEL,
            "input": batch,
        }

        r = requests.post(
            f"{API_BASE}/embeddings",
            headers=HEADERS,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()

        data = r.json()["data"]

        batch_embeddings = [d["embedding"] for d in data]
        all_embeddings.extend(batch_embeddings)

        print(f"Embedded {len(all_embeddings)} / {len(texts)}")

        time.sleep(0.5)

    return np.array(all_embeddings)



def build_prompt(problem):
    return f"""
Generate {N_TESTS_PER_PROBLEM} Python unit test functions for the problem below.

Requirements:
- Each test must be a standalone Python function
- Ordered from MOST IMPORTANT to LEAST IMPORTANT
- Vary test function names meaningfully
- Do NOT include explanations
- Separate tests by a blank line
- Output ONLY Python code.
- Do NOT wrap it in a code block.

Problem:
{problem["prompt"]}

Reference solution:
{problem["canonical_solution"]}
""".strip()


def split_tests(code_block):
    """
    Splits model output into individual test functions.
    Assumes functions start with 'def test'.
    """
    tests = []
    current = []

    for line in code_block.splitlines():
        if line.startswith("def test") and current:
            tests.append("\n".join(current).strip())
            current = []
        current.append(line)

    if current:
        tests.append("\n".join(current).strip())

    return tests


def main():

    problems = load_dataset("openai/openai_humaneval", split="test")
    assert isinstance(problems, Dataset)

    if TESTS_JSON.exists():
        with open(TESTS_JSON) as f:
            dataset = json.load(f)
    else:
        dataset = []

    existing_keys = {
        (item["problem_id"], item["test_index"])
        for item in dataset
    }

    print(f"Loaded {len(dataset)} existing tests")

    for pid, problem in tqdm(enumerate(problems), desc="HumanEval Problems"):
        expected = {(pid, i) for i in range(N_TESTS_PER_PROBLEM)}
        missing = expected - existing_keys

        if not missing:
            continue

        # print(f"Generating tests for {pid}")

        raw_code = chat_completion(
            [
                {"role": "system", "content": "Write precise Python unit tests."},
                {"role": "user", "content": build_prompt(problem)},
            ]
        )

        tests = split_tests(raw_code)

        if len(tests) != N_TESTS_PER_PROBLEM:
            print(
                f"WARNING: expected {N_TESTS_PER_PROBLEM} tests, "
                f"got {len(tests)} for {pid}"
            )

        for i, test_code in enumerate(tests):
            key = (pid, i)
            if key in existing_keys:
                continue

            dataset.append({
                "problem_id": pid,
                "test_index": i,
                "importance_rank": i + 1,
                "text": test_code,
            })

        with open(TESTS_JSON, "w") as f:
            json.dump(dataset, f, indent=2)

        time.sleep(0.5)

    print("All tests ready.")

    texts = [item["text"] for item in dataset]
    embeddings = embed_texts_chunked(texts)

    np.save(EMBED_NPY, embeddings)
    print(f"Saved embeddings: {embeddings.shape}")

if __name__ == "__main__":
    main()

