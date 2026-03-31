import pprint
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

API_BASE = "https://api.semanticscholar.org/graph/v1"
MIN_REFS = 40
TARGET_PAPERS = 1200
REQUEST_DELAY = 3.1
REQUEST_DELAY_WITH_KEY = 1
MAX_RETRIES = 3
CHECKPOINT_EVERY = 50
OUTPUT_DIR = Path("artifacts/litreview_corpus")

PAPER_FIELDS = "title,abstract,year,authors,referenceCount,citationCount,fieldsOfStudy,publicationTypes"
REF_FIELDS = "title,year,referenceCount,authors"

# Broad field queries for discovering diverse recent papers
FIELD_QUERIES = [
    "deep learning",
    "large language models",
    "computer vision",
    "reinforcement learning",
    "graph neural networks",
    "natural language processing",
    "robotics control",
    "drug discovery machine learning",
    "climate modeling",
    "computational biology",
    "recommender systems",
    "speech recognition",
    "autonomous driving",
    "medical image analysis",
    "quantum computing",
    "federated learning",
    "neural architecture search",
    "time series forecasting",
    "knowledge graphs",
    "multimodal learning",
]

SEED_YEAR_RANGE = "2022-2024"  # recent papers
SEEDS_PER_FIELD = 3  # top N papers per field query
MIN_SEED_REFS = 50  # seeds should have more refs than MIN_REFS


def _get(url: str, params: dict, api_key: str | None = None, delay: float = REQUEST_DELAY) -> dict | None:
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                time.sleep(delay)
                return r.json()
            if r.status_code == 429:
                wait = 2 ** (attempt + 2)
                log.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                time.sleep(delay)
                return None
            log.warning(f"HTTP {r.status_code} on attempt {attempt+1}: {r.text[:200]}")
            time.sleep(2 ** (attempt + 1))
        except requests.RequestException as e:
            log.warning(f"Request error attempt {attempt+1}: {e}")
            time.sleep(2 ** (attempt + 1))
    return None


def discover_seeds_for_field(
    query: str, 
    api_key: str | None = None, 
    delay: float = REQUEST_DELAY,
    n: int = SEEDS_PER_FIELD,
) -> list[dict]:
    """Search for recent papers in a field and return those with many references."""
    data = _get(
        f"{API_BASE}/paper/search",
        {
            "query": query,
            "fields": "referenceCount,title,year",
            "limit": 20,  # fetch more, filter locally
            "year": SEED_YEAR_RANGE,
        },
        api_key,
        delay,
    )
    if not data or not data.get("data"):
        return []
    
    # Filter and sort by referenceCount
    candidates = [
        p for p in data["data"]
        if (p.get("referenceCount") or 0) >= MIN_SEED_REFS
    ]
    candidates.sort(key=lambda p: p.get("referenceCount") or 0, reverse=True)
    
    return candidates[:n]


def resolve_seeds(api_key: str | None = None, delay: float = REQUEST_DELAY) -> list[str]:
    """Discover diverse seed papers across multiple fields."""
    all_seeds: dict[str, dict] = {}  # paperId -> paper info (dedup)
    
    log.info(f"Discovering seeds across {len(FIELD_QUERIES)} fields (year: {SEED_YEAR_RANGE})...")
    
    for query in FIELD_QUERIES:
        papers = discover_seeds_for_field(query, api_key, delay)
        for p in papers:
            pid = p.get("paperId")
            if pid and pid not in all_seeds:
                all_seeds[pid] = p
                log.info(f"  [{query[:20]}] {p.get('title', '?')[:50]} (refs={p.get('referenceCount')})")
    
    # Sort all discovered seeds by referenceCount and take top ones
    sorted_seeds = sorted(
        all_seeds.items(),
        key=lambda x: x[1].get("referenceCount") or 0,
        reverse=True,
    )
    
    # Return paper IDs
    seed_ids = [pid for pid, _ in sorted_seeds]
    log.info(f"Discovered {len(seed_ids)} unique seed papers")
    return seed_ids


def fetch_paper_with_refs(paper_id: str, api_key: str | None = None, delay: float = REQUEST_DELAY) -> dict | None:
    """Fetch a paper's metadata + its references."""
    fields = f"{PAPER_FIELDS},references.{REF_FIELDS.replace(',', ',references.')}"
    return _get(f"{API_BASE}/paper/{paper_id}", {"fields": fields}, api_key, delay)


def save_checkpoint(collected: dict, queue_state: list, seen: set, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "papers.jsonl", "w") as f:
        for pid, paper in collected.items():
            f.write(json.dumps({"paperId": pid, **paper}) + "\n")
    with open(path / "state.json", "w") as f:
        json.dump({"queue": queue_state, "seen": list(seen)}, f)
    log.info(f"Checkpoint: {len(collected)} papers saved")


def load_checkpoint(path: Path) -> tuple[dict, deque, set] | None:
    papers_path = path / "papers.jsonl"
    state_path = path / "state.json"
    if not papers_path.exists() or not state_path.exists():
        return None

    collected = {}
    with open(papers_path) as f:
        for line in f:
            obj = json.loads(line)
            pid = obj.pop("paperId")
            collected[pid] = obj

    with open(state_path) as f:
        state = json.load(f)

    queue = deque(state["queue"])
    seen = set(state["seen"])
    log.info(f"Resumed: {len(collected)} papers, {len(queue)} in queue, {len(seen)} seen")
    return collected, queue, seen


def qualifies(paper: dict) -> bool:
    ref_count = paper.get("referenceCount") or 0
    abstract = paper.get("abstract")
    return ref_count >= MIN_REFS and abstract and len(abstract.strip()) > 50


def extract_ref_metadata(ref: dict) -> dict | None:
    """Extract minimal reference metadata for steganography."""
    authors = ref.get("authors") or []
    if not authors or not ref.get("title") or not ref.get("year"):
        return None
    
    # Get first author's last name
    first_author = authors[0].get("name", "")
    last_name = first_author.split()[-1] if first_author else ""
    
    return {
        "paperId": ref.get("paperId"),
        "title": ref.get("title"),
        "year": ref.get("year"),
        "author_last_name": last_name,
        "author_text": f"{last_name} et al." if len(authors) > 1 else last_name,
    }


def strip_for_storage(paper: dict) -> dict:
    """Keep paper metadata, drop references (fetched separately)."""
    return {k: v for k, v in paper.items() if k != "references"}


def collect(api_key: str | None = None, resume: bool = True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    delay = REQUEST_DELAY_WITH_KEY if api_key else REQUEST_DELAY

    collected: dict[str, dict] = {}
    seen: set[str] = set()
    queue: deque[str] = deque()

    if resume:
        checkpoint = load_checkpoint(OUTPUT_DIR)
        if checkpoint:
            collected, queue, seen = checkpoint

    if not queue and not collected:
        seed_ids = resolve_seeds(api_key, delay)
        for pid in seed_ids:
            if pid not in seen:
                queue.append(pid)
                seen.add(pid)
        log.info(f"Enqueued {len(queue)} seed papers")

    papers_since_checkpoint = 0
    total_requests = 0

    log.info(f"Target: {TARGET_PAPERS} papers with {MIN_REFS}+ references")
    log.info(f"Starting BFS: {len(queue)} in queue, {len(collected)} collected")
    log.info(f"Request delay: {delay}s ({'with API key' if api_key else 'unauthenticated'})")

    while queue and len(collected) < TARGET_PAPERS:
        paper_id = queue.popleft()

        data = fetch_paper_with_refs(paper_id, api_key, delay)
        total_requests += 1

        if data is None:
            continue

        if qualifies(data) and paper_id not in collected:
            collected[paper_id] = strip_for_storage(data)
            papers_since_checkpoint += 1
            if len(collected) % 10 == 0:
                log.info(
                    f"[{len(collected)}/{TARGET_PAPERS}] "
                    f"queue={len(queue)} seen={len(seen)} reqs={total_requests} "
                    f"| {data.get('title', '?')[:50]}"
                )

        refs = data.get("references") or []
        new_enqueued = 0
        for ref in refs:
            rid = ref.get("paperId")
            if not rid or rid in seen:
                continue
            ref_count = ref.get("referenceCount") or 0
            if ref_count >= MIN_REFS:
                queue.append(rid)
                seen.add(rid)
                new_enqueued += 1

        if papers_since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(collected, list(queue), seen, OUTPUT_DIR)
            papers_since_checkpoint = 0

    save_checkpoint(collected, list(queue), seen, OUTPUT_DIR)
    log.info(f"Done: {len(collected)} papers collected ({total_requests} API requests)")

    with open(OUTPUT_DIR / "corpus.jsonl", "w") as f:
        for pid, paper in collected.items():
            f.write(json.dumps({"paperId": pid, **paper}) + "\n")
    log.info(f"Corpus written to {OUTPUT_DIR / 'corpus.jsonl'}")


def collect_references(api_key: str | None = None, resume: bool = True):
    """Fetch references for all papers in corpus.jsonl."""
    corpus_path = OUTPUT_DIR / "corpus.jsonl"
    refs_path = OUTPUT_DIR / "references.jsonl"
    
    if not corpus_path.exists():
        log.error(f"Corpus not found: {corpus_path}")
        log.error("Run 'collect' first to gather papers.")
        return
    
    delay = REQUEST_DELAY_WITH_KEY if api_key else REQUEST_DELAY
    
    # Load paper IDs from corpus
    paper_ids = []
    with open(corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            paper_ids.append(obj["paperId"])
    log.info(f"Found {len(paper_ids)} papers in corpus")
    
    # Load existing progress
    done_ids: set[str] = set()
    
    if resume and refs_path.exists():
        with open(refs_path) as f:
            for line in f:
                obj = json.loads(line)
                done_ids.add(obj["paperId"])
        log.info(f"Resumed: {len(done_ids)} papers already have references")
    
    # Fetch references for remaining papers
    pending = [pid for pid in paper_ids if pid not in done_ids]
    log.info(f"Fetching references for {len(pending)} papers")
    log.info(f"Request delay: {delay}s ({'with API key' if api_key else 'unauthenticated'})")
    
    # Open in append mode
    with open(refs_path, "a") as f:
        for i, paper_id in enumerate(pending):
            data = fetch_paper_with_refs(paper_id, api_key, delay)
            
            if data is None:
                log.warning(f"  Failed to fetch {paper_id}")
                refs = []
            else:
                raw_refs = data.get("references") or []
                refs = [
                    r for r in (extract_ref_metadata(ref) for ref in raw_refs)
                    if r is not None
                ]
            
            # Write immediately
            f.write(json.dumps({"paperId": paper_id, "references": refs}) + "\n")
            f.flush()
            
            if (i + 1) % 10 == 0:
                log.info(f"[{i + 1}/{len(pending)}] {paper_id[:12]}... ({len(refs)} refs)")
    
    log.info(f"Done: references written to {refs_path}")


def retry_empty_references(api_key: str | None = None, verbose: bool = False):
    """Retry fetching references for papers that have 0 refs."""
    corpus_path = OUTPUT_DIR / "corpus.jsonl"
    refs_path = OUTPUT_DIR / "references.jsonl"
    
    if not refs_path.exists():
        log.error(f"References file not found: {refs_path}")
        return
    
    delay = REQUEST_DELAY_WITH_KEY if api_key else REQUEST_DELAY
    
    # Load current references and find empty ones
    all_refs: dict[str, list] = {}
    empty_ids: list[str] = []
    
    with open(refs_path) as f:
        for line in f:
            obj = json.loads(line)
            pid = obj["paperId"]
            refs = obj["references"]
            all_refs[pid] = refs
            if len(refs) == 0:
                empty_ids.append(pid)
    
    log.info(f"Found {len(empty_ids)} papers with 0 references out of {len(all_refs)} total")
    
    if not empty_ids:
        log.info("Nothing to retry!")
        return
    
    # Retry fetching
    fixed = 0
    still_empty = 0
    
    for i, paper_id in enumerate(empty_ids):
        log.info(f"\n[{i+1}/{len(empty_ids)}] Retrying {paper_id}")
        
        # Fetch with explicit field request
        fields = f"{PAPER_FIELDS},references.{REF_FIELDS.replace(',', ',references.')}"
        
        if verbose:
            log.info(f"  Request URL: {API_BASE}/paper/{paper_id}")
            log.info(f"  Fields: {fields}")
        
        data = _get(f"{API_BASE}/paper/{paper_id}", {"fields": fields}, api_key, delay)
        
        if data is None:
            log.warning(f"  API returned None (request failed)")
            still_empty += 1
            continue
        
        if verbose:
            log.info(f"  Response keys: {list(data.keys())}")
            log.info(f"  Title: {data.get('title', 'N/A')[:60]}")
            log.info(f"  referenceCount field: {data.get('referenceCount')}")
            refs_raw = data.get("references")
            log.info(f"  references field type: {type(refs_raw)}")
            if refs_raw:
                log.info(f"  references count: {len(refs_raw)}")
                if len(refs_raw) > 0:
                    log.info(f"  First ref keys: {list(refs_raw[0].keys()) if isinstance(refs_raw[0], dict) else refs_raw[0]}")
                    log.info(f"  First ref sample: {refs_raw[0]}")
            else:
                log.info(f"  references field is empty/None: {refs_raw}")
        
        raw_refs = data.get("references") or []
        
        # Try to extract with logging
        extracted = []
        skipped_reasons: dict[str, int] = {"no_authors": 0, "no_title": 0, "no_year": 0}
        
        for ref in raw_refs:
            authors = ref.get("authors") or []
            title = ref.get("title")
            year = ref.get("year")
            
            if not authors:
                skipped_reasons["no_authors"] += 1
                continue
            if not title:
                skipped_reasons["no_title"] += 1
                continue
            if not year:
                skipped_reasons["no_year"] += 1
                continue
            
            first_author = authors[0].get("name", "")
            last_name = first_author.split()[-1] if first_author else ""
            
            extracted.append({
                "paperId": ref.get("paperId"),
                "title": title,
                "year": year,
                "author_last_name": last_name,
                "author_text": f"{last_name} et al." if len(authors) > 1 else last_name,
            })
        
        if verbose and raw_refs:
            log.info(f"  Raw refs: {len(raw_refs)}, Extracted: {len(extracted)}")
            log.info(f"  Skipped reasons: {skipped_reasons}")
        
        if extracted:
            all_refs[paper_id] = extracted
            fixed += 1
            log.info(f"  Fixed! Now has {len(extracted)} references")
        else:
            still_empty += 1
            log.info(f"  Still empty after retry")
    
    # Rewrite the entire references file
    log.info(f"\nRewriting {refs_path}...")
    with open(refs_path, "w") as f:
        for pid, refs in all_refs.items():
            f.write(json.dumps({"paperId": pid, "references": refs}) + "\n")
    
    log.info(f"Done: Fixed {fixed}, still empty {still_empty}")


def clean(min_extracted_refs: int = MIN_REFS, target: int = TARGET_PAPERS):
    """Remove papers with too few extractable refs from both files. Report gap."""
    corpus_path = OUTPUT_DIR / "corpus.jsonl"
    refs_path = OUTPUT_DIR / "references.jsonl"

    if not corpus_path.exists() or not refs_path.exists():
        log.error("corpus.jsonl or references.jsonl not found")
        return

    # Load references and find usable paper IDs
    ref_counts: dict[str, int] = {}
    with open(refs_path) as f:
        for line in f:
            obj = json.loads(line)
            ref_counts[obj["paperId"]] = len(obj["references"])

    usable_ids = {pid for pid, count in ref_counts.items() if count >= min_extracted_refs}
    dropped = len(ref_counts) - len(usable_ids)

    log.info(f"References file: {len(ref_counts)} papers total")
    log.info(f"  Usable (>= {min_extracted_refs} extracted refs): {len(usable_ids)}")
    log.info(f"  Dropping: {dropped}")

    # Filter corpus.jsonl
    kept_corpus = []
    with open(corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj["paperId"] in usable_ids:
                kept_corpus.append(line)

    with open(corpus_path, "w") as f:
        f.writelines(kept_corpus)
    log.info(f"  corpus.jsonl: {len(kept_corpus)} papers retained")

    # Filter references.jsonl
    kept_refs = []
    with open(refs_path) as f_in:
        for line in f_in:
            obj = json.loads(line)
            if obj["paperId"] in usable_ids:
                kept_refs.append(line)

    with open(refs_path, "w") as f:
        f.writelines(kept_refs)
    log.info(f"  references.jsonl: {len(kept_refs)} papers retained")

    gap = target - len(usable_ids)
    if gap > 0:
        log.info(f"\n  Gap: need {gap} more papers to reach {target}")
        log.info(f"  Run: python {__file__} backfill --target {target}")
    else:
        log.info(f"\n  Already at {len(usable_ids)} >= {target} target. Done!")


def backfill(api_key: str | None = None, target: int = TARGET_PAPERS, resume: bool = True):
    """Resume BFS, verifying actual extractable refs before accepting papers.

    Reads existing corpus.jsonl + references.jsonl as the starting set,
    then resumes from the BFS checkpoint (state.json) to fill the gap.
    Writes both files in one pass per accepted paper.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    delay = REQUEST_DELAY_WITH_KEY if api_key else REQUEST_DELAY

    corpus_path = OUTPUT_DIR / "corpus.jsonl"
    refs_path = OUTPUT_DIR / "references.jsonl"

    # Load existing accepted papers
    existing_ids: set[str] = set()
    if corpus_path.exists():
        with open(corpus_path) as f:
            for line in f:
                existing_ids.add(json.loads(line)["paperId"])
    log.info(f"Existing accepted papers: {len(existing_ids)}")

    if len(existing_ids) >= target:
        log.info(f"Already at {len(existing_ids)} >= {target}. Nothing to do.")
        return

    # Load BFS state
    state_path = OUTPUT_DIR / "state.json"
    if resume and state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        queue = deque(state["queue"])
        seen = set(state["seen"])
        log.info(f"Resumed BFS: {len(queue)} in queue, {len(seen)} seen")
    else:
        log.error("No BFS state found. Run 'collect' first or check state.json.")
        return

    # Make sure all existing papers are in seen set
    seen.update(existing_ids)

    accepted = len(existing_ids)
    total_requests = 0
    papers_since_checkpoint = 0
    skipped_no_refs = 0

    log.info(f"Target: {target} papers, need {target - accepted} more")
    log.info(f"Request delay: {delay}s ({'with API key' if api_key else 'unauthenticated'})")

    # Open both files in append mode
    with open(corpus_path, "a") as f_corpus, open(refs_path, "a") as f_refs:
        while queue and accepted < target:
            paper_id = queue.popleft()

            data = fetch_paper_with_refs(paper_id, api_key, delay)
            total_requests += 1

            if data is None:
                continue

            # Extract actual refs
            raw_refs = data.get("references") or []
            extracted_refs = [
                r for r in (extract_ref_metadata(ref) for ref in raw_refs)
                if r is not None
            ]

            # Only accept if metadata qualifies AND we can actually extract enough refs
            if (
                qualifies(data)
                and paper_id not in existing_ids
                and len(extracted_refs) >= MIN_REFS
            ):
                paper_meta = strip_for_storage(data)
                f_corpus.write(json.dumps({"paperId": paper_id, **paper_meta}) + "\n")
                f_corpus.flush()
                f_refs.write(json.dumps({"paperId": paper_id, "references": extracted_refs}) + "\n")
                f_refs.flush()

                existing_ids.add(paper_id)
                accepted += 1
                papers_since_checkpoint += 1

                if accepted % 10 == 0:
                    log.info(
                        f"[{accepted}/{target}] "
                        f"queue={len(queue)} seen={len(seen)} reqs={total_requests} "
                        f"skipped_no_refs={skipped_no_refs} "
                        f"| {data.get('title', '?')[:50]}"
                    )
            elif qualifies(data) and len(extracted_refs) < MIN_REFS:
                skipped_no_refs += 1

            # Enqueue promising refs for BFS
            for ref in raw_refs:
                rid = ref.get("paperId")
                if not rid or rid in seen:
                    continue
                ref_count = ref.get("referenceCount") or 0
                if ref_count >= MIN_REFS:
                    queue.append(rid)
                    seen.add(rid)

            # Save BFS state periodically
            if papers_since_checkpoint >= CHECKPOINT_EVERY:
                with open(state_path, "w") as f_state:
                    json.dump({"queue": list(queue), "seen": list(seen)}, f_state)
                log.info(f"  BFS checkpoint saved")
                papers_since_checkpoint = 0

    # Final BFS state save
    with open(state_path, "w") as f_state:
        json.dump({"queue": list(queue), "seen": list(seen)}, f_state)

    log.info(
        f"Done: {accepted} total papers ({total_requests} requests, "
        f"{skipped_no_refs} skipped for no extractable refs)"
    )


def diagnose_paper(paper_id: str, api_key: str | None = None):
    """Fetch a single paper and show full debug info."""
    delay = REQUEST_DELAY_WITH_KEY if api_key else REQUEST_DELAY
    fields = f"{PAPER_FIELDS},references.{REF_FIELDS.replace(',', ',references.')}"
    
    log.info(f"Diagnosing paper: {paper_id}")
    log.info(f"Request URL: {API_BASE}/paper/{paper_id}")
    log.info(f"Fields: {fields}")
    
    data = _get(f"{API_BASE}/paper/{paper_id}", {"fields": fields}, api_key, delay)
    
    if data is None:
        log.error("API returned None!")
        return
    
    log.info(f"\n=== Paper Info ===")
    log.info(f"Title: {data.get('title')}")
    log.info(f"Year: {data.get('year')}")
    log.info(f"referenceCount: {data.get('referenceCount')}")
    
    refs = data.get("references")
    log.info(f"\n=== References ===")
    log.info(f"Type: {type(refs)}")
    log.info(f"Count: {len(refs) if refs else 0}")
    
    if refs:
        log.info(f"\nFirst 3 references (raw):")
        for i, ref in enumerate(refs[:3]):
            log.info(f"  [{i}] {json.dumps(ref, indent=4)}")
        
        # Check what extract_ref_metadata would do
        log.info(f"\nExtraction test:")
        for i, ref in enumerate(refs[:3]):
            result = extract_ref_metadata(ref)
            log.info(f"  [{i}] -> {result}")
    else:
        pprint.pprint(data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "refs", "retry", "diagnose", "clean", "backfill"], 
                        help="'collect' papers, 'refs' to fetch references, 'retry' empty refs, "
                             "'clean' drop papers with 0 refs, 'backfill' resume BFS with ref verification, "
                             "'diagnose' a single paper")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--target", type=int, default=TARGET_PAPERS)
    parser.add_argument("--min-refs", type=int, default=MIN_REFS)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output for retry/diagnose")
    parser.add_argument("--paper-id", default=None, help="Paper ID for diagnose command")
    args = parser.parse_args()

    # Load API key from environment
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        log.info("API key found in environment")
    else:
        log.warning("No API key found - using unauthenticated rate limits (slow)")

    if args.target != TARGET_PAPERS:
        TARGET_PAPERS = args.target
    if args.min_refs != MIN_REFS:
        MIN_REFS = args.min_refs

    if args.command == "collect":
        collect(api_key=api_key, resume=not args.no_resume)
    elif args.command == "refs":
        collect_references(api_key=api_key, resume=not args.no_resume)
    elif args.command == "retry":
        retry_empty_references(api_key=api_key, verbose=args.verbose)
    elif args.command == "clean":
        clean(min_extracted_refs=args.min_refs, target=args.target)
    elif args.command == "backfill":
        backfill(api_key=api_key, target=args.target, resume=not args.no_resume)
    elif args.command == "diagnose":
        if not args.paper_id:
            log.error("--paper-id required for diagnose command")
        else:
            diagnose_paper(args.paper_id, api_key=api_key)
