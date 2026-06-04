# LitReview reference corpus

Reference dataset for the **LitReview** steganography system. Unlike the story and
topicQA systems (which require live LLM services to reproduce their inputs), this
system runs against a fixed corpus of papers and their citations, so we ship that
corpus here directly.

## Contents

- `references/corpus.jsonl` — one paper per line: `paperId`, `title`, `abstract`,
  `year`, `authors`. (1200 papers.)
- `references/references.jsonl` — one record per line keyed by `paperId`, holding
  that paper's cited `references`.
- `references/papers.jsonl` — the full paper-id pool used to expand LitReview
  prompts.
- `references/state.json` — scraper checkpoint (queue / seen set) for resuming a run.
- `semantic_scholar_requests.py` — the scraper that produced the corpus via the
  [Semantic Scholar Graph API](https://api.semanticscholar.org/graph/v1).

The corpus is loaded by `systems.core.litreview.load_corpus`, which joins
`corpus.jsonl` + `references.jsonl` and keeps papers that have both an abstract and
references (~1191 of 1200).

## Regenerating

```bash
python data/litreview/semantic_scholar_requests.py
```

The scraper reads `SEMANTIC_SCHOLAR_API_KEY` from the repo-root `.env` automatically
(for higher rate limits); without it, it falls back to unauthenticated limits. Output
is written into `references/` (resolved relative to the script).

## Path access

Consumers resolve these files through `systems.paths`, not hardcoded strings:

- `litreview_references()` → `(corpus.jsonl, references.jsonl)`
- `litreview_papers()` → `papers.jsonl`

Set the `DATA_DIR` environment variable to relocate the data root away from
`<repo>/data`.
