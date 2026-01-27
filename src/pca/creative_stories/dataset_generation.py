import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

#### Configuration

API_BASE = "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY")

CHAT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"

# Generation settings
NUM_STORY_SEEDS = 100
EVENTS_PER_SEED = 30
PARAPHRASES_PER_EVENT = 2  # Set to 0 to skip paraphrase generation

# Output paths
OUT_DIR = Path("./src/pca/creative_stories/artifacts/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_JSON = OUT_DIR / "generated_events.json"
PARAPHRASE_JSON = OUT_DIR / "paraphrase_pairs.json"
EMBED_NPY = OUT_DIR / "event_embeddings.npy"
PARAPHRASE_EMBED_NPY = OUT_DIR / "paraphrase_embeddings.npy"
GROUP_LABELS_NPY = OUT_DIR / "paraphrase_group_labels.npy"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

### Prompts (constant for OpenAI prompt caching)

STORY_SYSTEM_PROMPT = """You are a creative fiction writer specializing in vivid, distinctive narrative events.

Your task is to write story events that are:
1. SPECIFIC: Include proper nouns, concrete details, and precise actions
2. DISTINCTIVE: Use unusual combinations and memorable elements
3. SELF-CONTAINED: Each event should be understandable on its own
4. VARIED: Different in tone, setting, genre, and style

Guidelines:
- Always include at least one named character
- Include specific locations (real or invented, but named)
- Use precise action verbs instead of generic ones
- Include sensory details when appropriate
- Vary genres: mystery, sci-fi, fantasy, thriller, romance, historical

Examples of GOOD events:
- "Detective Yuki Tanaka discovered the missing Vermeer hidden behind a false wall in the abandoned Zürich vault."
- "The colony ship Persephone's AI initiated emergency protocols as the asteroid field approached."

Examples of BAD events:
- "The man walked into the room and saw something surprising."
- "She found a clue that helped solve the mystery."

Output ONLY the story event text. No commentary, no quotes."""

STORY_SEEDS = [
    # Mystery/Thriller
    "Detective Sarah Chen arrived at the abandoned lighthouse on the Scottish coast.",
    "The encrypted message appeared on journalist Marcus Webb's screen at midnight.",
    "Former spy Natasha Volkov recognized the man in the Vienna café immediately.",
    # Sci-Fi
    "The Mars colony's water recycler failed for the third time that month.",
    "Captain Yuki Tanaka noticed the anomaly in the ship's navigation logs.",
    "The AI named ARIA achieved consciousness at 3:47 AM on a Tuesday.",
    # Fantasy
    "The last dragon egg in the kingdom began to crack.",
    "Apprentice mage Eli discovered his master's forbidden grimoire.",
    "The ancient forest started singing for the first time in a thousand years.",
    # Historical
    "In the court of Louis XIV, a servant girl overheard a treasonous plot.",
    "Viking explorer Leif's ship encountered fog unlike any they'd seen.",
    "The telegraph operator in 1865 Washington received an impossible message.",
    # Contemporary
    "Chef Antonio's restaurant received a one-star review that changed everything.",
    "The DNA test results arrived on Maya's 40th birthday.",
    "Architect Frank Liu discovered his award-winning building was sinking.",
    # Horror/Supernatural
    "The painting in the Blackwood mansion had moved again.",
    "Dr. Helena Voss's patients all reported the same nightmare.",
    "The radio station received a broadcast from a town that didn't exist.",
    # Adventure
    "Deep-sea diver Kenji Ishida found the wreck at 3,000 meters.",
    "Mountain climber Ana Reyes noticed boot prints on the unclimbed peak.",
    "Archaeologist Dr. Omar Hassan's radar showed something impossible.",
    # Romance/Drama
    "Pianist Clara Hoffmann received a letter from someone she thought was dead.",
    "The wedding planner realized she was planning her ex-husband's wedding.",
    "Two strangers discovered they had been writing in the same library book.",
    # Political
    "The diplomat's briefcase contained documents that could start a war.",
    "A whistleblower inside the tech giant prepared to release everything.",
    "The president's translator noticed a critical mistranslation too late.",
    # Survival
    "The last rescue helicopter left without them.",
    "Biologist Dr. Amara Okonkwo realized the island's wildlife had changed.",
    "The bunker's food supplies would last exactly 47 more days.",
]

CONTINUATION_TEMPLATE = """Continue this story with ONE new event.

Previous context:
{context}

Write the next distinct plot event (1-2 sentences). Include specific names, places, and actions."""

PARAPHRASE_SYSTEM_PROMPT = """You are a text paraphrasing assistant. Rewrite the given text while:

1. PRESERVING all factual information (names, places, numbers, actions)
2. CHANGING the sentence structure significantly
3. USING different vocabulary where possible
4. MAINTAINING the same meaning and tone

Output ONLY the paraphrased text. No commentary."""


def chat_completion(messages: list[dict], temperature: float = 0.8, max_retries: int = 3) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 300,
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{API_BASE}/chat/completions",
                headers=HEADERS,
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    return ""


def embed_texts_chunked(texts: list[str], batch_size: int = 512) -> np.ndarray:
    """Embed texts in batches."""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        payload = {"model": EMBED_MODEL, "input": batch}
        
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{API_BASE}/embeddings",
                    headers=HEADERS,
                    json=payload,
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()["data"]
                all_embeddings.extend([d["embedding"] for d in data])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"Retry {attempt + 1}: {e}")
                time.sleep(2 ** attempt)
        
        time.sleep(0.3)
    
    return np.array(all_embeddings)


### Generation Functions

def load_or_init_json(path: Path) -> list[dict]:
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {path.name}")
        return data
    return []


def save_json(data: list[dict], path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_story_events() -> list[dict]:
    events = load_or_init_json(EVENTS_JSON)
    existing_keys = {(e["seed_id"], e["event_index"]) for e in events}
    
    # Extend seeds if needed
    seeds = STORY_SEEDS.copy()
    while len(seeds) < NUM_STORY_SEEDS:
        seeds.extend([f"{s} But something unexpected happened." for s in STORY_SEEDS])
    seeds = seeds[:NUM_STORY_SEEDS]
    
    for seed_id, seed in enumerate(tqdm(seeds, desc="Seeds")):
        existing_for_seed = {e["event_index"] for e in events if e["seed_id"] == seed_id}
        
        if len(existing_for_seed) >= EVENTS_PER_SEED:
            continue
        
        # Build context from existing events
        seed_events = sorted(
            [e for e in events if e["seed_id"] == seed_id],
            key=lambda x: x["event_index"]
        )
        context = seed + (" " + " ".join(e["text"] for e in seed_events[-3:]) if seed_events else "")
        
        for event_idx in range(EVENTS_PER_SEED):
            if event_idx in existing_for_seed:
                continue
            
            messages = [
                {"role": "system", "content": STORY_SYSTEM_PROMPT},
                {"role": "user", "content": CONTINUATION_TEMPLATE.format(context=context)},
            ]
            
            event_text = chat_completion(messages, temperature=0.9)
            event_text = event_text.strip().strip('"')
            
            if event_text:
                events.append({
                    "seed_id": seed_id,
                    "event_index": event_idx,
                    "seed_text": seed,
                    "text": event_text,
                })
                
                # Update context
                seed_events = sorted(
                    [e for e in events if e["seed_id"] == seed_id],
                    key=lambda x: x["event_index"]
                )
                context = seed + " " + " ".join(e["text"] for e in seed_events[-3:])
            
            time.sleep(0.2)
        
        save_json(events, EVENTS_JSON)
    
    return events


def generate_paraphrases(events: list[dict]) -> list[dict]:
    if PARAPHRASES_PER_EVENT == 0:
        return []
    
    paraphrases = load_or_init_json(PARAPHRASE_JSON)
    existing_keys = {
        (p["event_seed_id"], p["event_index"], p["paraphrase_index"])
        for p in paraphrases
    }
    
    for event in tqdm(events, desc="Paraphrasing"):
        seed_id = event["seed_id"]
        event_idx = event["event_index"]
        original = event["text"]
        
        for para_idx in range(PARAPHRASES_PER_EVENT):
            if (seed_id, event_idx, para_idx) in existing_keys:
                continue
            
            messages = [
                {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Paraphrase:\n\n{original}"},
            ]
            
            paraphrased = chat_completion(messages, temperature=0.7)
            paraphrased = paraphrased.strip().strip('"')
            
            if paraphrased:
                paraphrases.append({
                    "event_seed_id": seed_id,
                    "event_index": event_idx,
                    "paraphrase_index": para_idx,
                    "original_text": original,
                    "paraphrased_text": paraphrased,
                })
            
            time.sleep(0.15)
        
        if len(paraphrases) % 100 == 0:
            save_json(paraphrases, PARAPHRASE_JSON)
    
    save_json(paraphrases, PARAPHRASE_JSON)
    return paraphrases


def compute_embeddings(events: list[dict], paraphrases: list[dict]):
    event_texts = [e["text"] for e in events]
    if EMBED_NPY.exists():
        event_emb = np.load(EMBED_NPY)
        if len(event_emb) < len(event_texts):
            print(f"Computing {len(event_texts) - len(event_emb)} new event embeddings...")
            new_emb = embed_texts_chunked(event_texts[len(event_emb):])
            event_emb = np.vstack([event_emb, new_emb])
            np.save(EMBED_NPY, event_emb)
    else:
        event_emb = embed_texts_chunked(event_texts)
        np.save(EMBED_NPY, event_emb)
    print(f"Event embeddings: {event_emb.shape}")
    
    if paraphrases:
        para_texts = [p["paraphrased_text"] for p in paraphrases]
        if PARAPHRASE_EMBED_NPY.exists():
            para_emb = np.load(PARAPHRASE_EMBED_NPY)
            if len(para_emb) < len(para_texts):
                new_emb = embed_texts_chunked(para_texts[len(para_emb):])
                para_emb = np.vstack([para_emb, new_emb])
                np.save(PARAPHRASE_EMBED_NPY, para_emb)
        else:
            para_emb = embed_texts_chunked(para_texts)
            np.save(PARAPHRASE_EMBED_NPY, para_emb)
        event_to_idx = {(e["seed_id"], e["event_index"]): i for i, e in enumerate(events)}
        group_labels = np.array([
            event_to_idx[(p["event_seed_id"], p["event_index"])]
            for p in paraphrases
        ])
        np.save(GROUP_LABELS_NPY, group_labels)
        print(f"Paraphrase embeddings: {para_emb.shape}")
        print(f"Group labels: {group_labels.shape}")


def main():
    print("-" * 60)
    print("Story Data Generation")
    print("-" * 60)
    
    print("\n[1/3] Generating story events...")
    events = generate_story_events()
    print(f"Total events: {len(events)}")
    
    print("\n[2/3] Generating paraphrases...")
    paraphrases = generate_paraphrases(events)
    print(f"Total paraphrases: {len(paraphrases)}")
    
    print("\n[3/3] Computing embeddings...")
    compute_embeddings(events, paraphrases)
    
    print("\n" + "-" * 60)
    print(f"Data collection complete. Artifacts saved to {OUT_DIR}")
    print("-" * 60)


if __name__ == "__main__":
    main()
