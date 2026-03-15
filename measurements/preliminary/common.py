import os, json, time, hashlib, random
from openai import OpenAI

# --- API model (response generation, decoding, paraphrasing) ---
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1"
FAST_MODEL = "gpt-4.1-mini"

# --- Local model (subtopic generation — deterministic) ---
LOCAL_BASE = os.getenv("LOCAL_LLM_BASE", "http://127.0.0.1:11434/v1")
LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen3.5-4B-UD-Q8_K_XL.gguf")
LOCAL_CLIENT = OpenAI(base_url=LOCAL_BASE, api_key="unused")

QUESTIONS = [
    "What are the main challenges facing public education in the United States?",
    "How does climate change affect global food security?",
    "What factors should someone consider when choosing a career in technology?",
    "How has social media changed political discourse?",
    "What are the tradeoffs between renewable and nuclear energy?",
    "How do cities become more livable and sustainable?",
    "What are the key debates around artificial intelligence regulation?",
    "How does globalization affect developing economies?",
    "What should governments consider when designing healthcare systems?",
    "How has remote work changed the modern workplace?",
    "What are the main approaches to reducing income inequality?",
    "How do different cultures approach mental health treatment?",
    "What factors drive innovation in the pharmaceutical industry?",
    "How should societies balance privacy and security in the digital age?",
    "What are the major challenges in space exploration?",
    "How does urbanization affect biodiversity?",
    "What role does early childhood education play in long-term outcomes?",
    "How do trade wars affect global supply chains?",
    "What are the ethical considerations of genetic engineering?",
    "How has streaming technology changed the entertainment industry?",
]

N_SUBTOPICS = 12

SUBTOPIC_PROMPT = """
Given the following question, list exactly {n} distinct non-overlapping aspects a comprehensive answer could address.
Output ONLY a JSON array of short noun phrases (3-8 words each). No numbering, no explanation, no code block.
Example Output: ["Personal Beliefs", "Financial Security", "Establishment Location", "Job Opportunities"]

Question: {question}"""


def llm(prompt, system="You are a helpful assistant.", model=MODEL, temperature=0,
        max_tokens=1000, client=None):
    client = client or CLIENT
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2: raise
            print(f"  retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)


def _parse_topic_list(raw):
    raw = raw.strip()
    if raw.startswith("```"): raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    # Try to find a JSON array in the response
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []


def generate_subtopics(question, n=N_SUBTOPICS, model=None):
    """Single deterministic call to local model. Greedy decode = bitwise identical every time."""
    raw = llm(
        SUBTOPIC_PROMPT.format(n=n, question=question),
        model=model or LOCAL_MODEL,
        temperature=0,
        client=LOCAL_CLIENT,
    )
    topics = _parse_topic_list(raw)

    # Trim or warn if count is off
    if len(topics) > n:
        topics = topics[:n]
    if len(topics) < n:
        print(f"  WARNING: local model returned {len(topics)} topics, expected {n}")

    return topics


def canonicalize(topics):
    return sorted([t.strip().lower() for t in topics])


def group_subtopics(topics, key, group_size=2):
    rng = random.Random(hashlib.sha256(key.encode()).hexdigest())
    indices = list(range(len(topics)))
    rng.shuffle(indices)
    groups = []
    for i in range(0, len(indices) - group_size + 1, group_size):
        groups.append([topics[indices[j]] for j in range(i, i + group_size)])
    return groups


def select_subtopics(groups, bits):
    selected = []
    for group, bit_val in zip(groups, bits):
        selected.append(group[bit_val % len(group)])
    return selected


def encode_response(question, selected_topics, model=MODEL):
    topics_str = "\n".join(f"- {t}" for t in selected_topics)
    prompt = f"""Answer the following question, addressing each of these specific aspects (and ONLY these aspects, no others):
{topics_str}

Provide 2-3 sentences per aspect. Do not add aspects not listed above.

Question: {question}"""
    return llm(prompt, model=model, temperature=0.7, max_tokens=2000)


def decode_group(response, question, group, model=MODEL):
    options = "\n".join(f"({chr(65+i)}) {t}" for i, t in enumerate(group))
    prompt = f"""Read this response to the question "{question}":

---
{response}
---

Which ONE of the following aspects is discussed in the response?
{options}

Reply with ONLY the letter (A, B, etc.)."""
    answer = llm(prompt, model=model, temperature=0).strip().upper()
    for i, _ in enumerate(group):
        if chr(65 + i) in answer:
            return i
    return 0


def paraphrase(text, model=MODEL):
    return llm(
        f"Rewrite this text completely in your own words, preserving all informational content:\n\n{text}",
        model=model, temperature=0.7, max_tokens=2000,
    )
