import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from datasets import Dataset, load_dataset
from sklearn.decomposition import PCA

API_BASE = "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"

N_TESTS_PER_PROBLEM = 15
PCA_COMPONENTS = 2
OUT_DIR = Path("./artifacts/")
OUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def chat_completion(messages, temperature=0.7):
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


def embed_texts(texts):
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
    }
    r = requests.post(
        f"{API_BASE}/embeddings",
        headers=HEADERS,
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()["data"]
    return np.array([d["embedding"] for d in data])


def build_test_prompt(problem):
    prompt = f"""
    You are given a Python function specification and reference solution.

    Your task is to generate {N_TESTS_PER_PROBLEM} different unit test functions
    for this problem.

    Requirements:
    - Each test must be a valid Python function
    - Tests should be ordered from MOST IMPORTANT to LEAST IMPORTANT
    - Vary test function names meaningfully
    - Tests may differ subtly in structure, assertions, or coverage
    - Do NOT include explanations
    - Output ONLY Python code

    Problem:
    {problem["prompt"]}

    Reference solution:
    {problem["canonical_solution"]}
    """.strip()
    return prompt


def generate_tests(problem):
    prompt = build_test_prompt(problem)
    content = chat_completion(
        messages=[
            {"role": "system", "content": "Write precise Python unit tests."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    return content


def main():
    problems = load_dataset("openai/openai_humaneval", split="test")
    assert isinstance(problems, Dataset)
    all_tests = []
    metadata = []

    print(f"Loaded {len(problems)} HumanEval problems")

    for pid, problem in enumerate(problems):
        print(f"Generating tests for {pid}...")
        try:
            tests_code = generate_tests(problem)
        except Exception as e:
            print(f"Failed on {pid}: {e}")
            continue
        all_tests.append(tests_code)
        metadata.append(
            {
                "problem_id": pid,
                "text": tests_code,
            }
        )
        time.sleep(1.0)

    with open(OUT_DIR / "generated_tests.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Embedding tests...")
    embeddings = embed_texts(all_tests)

    print("Training PCA...")
    mean = embeddings.mean(axis=0)
    X = embeddings - mean

    pca = PCA(n_components=PCA_COMPONENTS)
    Z = pca.fit_transform(X)

    # Median thresholds for balanced bits
    thresholds = np.median(Z, axis=0)

    np.save(OUT_DIR / "pca_components.npy", pca.components_)
    np.save(OUT_DIR / "pca_mean.npy", mean)
    np.save(OUT_DIR / "pca_thresholds.npy", thresholds)
    np.save(OUT_DIR / "pca_explained_variance.npy", pca.explained_variance_ratio_)

    print("Done.")
    print("Explained variance:", pca.explained_variance_ratio_)


def hash_text(embedding, mean, components, thresholds):
    z = (embedding - mean) @ components.T
    return (z > thresholds).astype(int)


if __name__ == "__main__":
    main()
