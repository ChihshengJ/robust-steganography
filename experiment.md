# Experiment Pipeline: LLM Steganography Paper

## Context

We're building experiments for a paper on LLM steganography. We have three steganography systems that encode secret bits into LLM-generated text. The core contribution is "discrete combinatorial selection with forced-choice decoding" — encoding lives in *which* semantic content the model chooses to include, not in token-level or embedding-level signals.

**Three systems:**

1. **TopicQASystem** — Encodes bits by selecting subtopics from key-permuted groups. Constructor: `TopicQASystem(client, error_correction, local_client, local_model, n_subtopics=12, group_size=2, response_model="gpt-4.1", decoder_model="gpt-4.1", key="default", encoder=None, response_temperature=0.7)`. API: `hide_message(data: list[int], seed: str)` where seed is the question string; `recover_message(stego_text: str)` — requires `_question` and `_error_encoded_length` to be set (set automatically by `hide_message`). Capacity: ~6 bits at n_subtopics=12, group_size=2.
2. **StorySlotSystem** (StorySystemV2) — Encodes bits by selecting A/B alternatives for story plot slots. Constructor: `StorySystemV2(client, error_correction, local_client, local_model, n_slots=18, encoder)`. API: `hide_message(data: list[int], seed: str)` where seed is the story premise; `recover_message(stego_text: str)`. Capacity: ~18 bits.
3. **LitReviewSystemV2** — Encodes bits by selecting references whose SHA-256 hash on (author_last_name, year) matches target bits. Constructor: `LitReviewSystemV2(client, error_correction, corpus=None, model="gpt-4.1", encoder=None)`. API: `hide_message(data: list[int], seed: int)` where seed is the corpus index; `recover_message(stego_text: str)` — requires `_error_encoded_length` set by prior `hide_message`. Capacity: ~20 bits.

**Common dependencies:**

- `BypassEncoder` — identity encoder, passes through ECC but skips character encoding. All systems use this for controlling the bits encoded.
- `RepetitionCode(n)` — ECC that repeats each bit n times. Use `RepetitionCode(1)` (no ECC) for all data generation.
- OpenAI client for GPT-4.1 (generation, decoding, paraphrasing).
- Local Qwen client via llama.cpp at `http://127.0.0.1:11434/v1` for deterministic subtopic/slot generation.
- Attack classes: `ParaphraseAttack(client, model, temperature)`, `TranslationAttack(client, model, temperature)`, `SynonymAttack(method="wordnet")`, `NGramShuffleAttack(model, n)`. All callable as `attack(text, tampering_rate, local_bool)`.

**Critical state management note:** `recover_message` depends on internal state set by `hide_message` (`_question` for TopicQA, `_error_encoded_length` for all systems). When separating generation from decoding (e.g., decode an attacked text hours later), you must restore this state. Save it alongside each generated text in the JSON output.

------

## Task Overview

Build a data generation and experiment pipeline that minimizes redundant API calls. The pipeline has 5 phases. All intermediate artifacts are saved as JSONL files so each phase can run independently and subsequent phases reuse prior outputs.

**Output directory structure:**

```
data/experiments/
├── phase1_texts/
│   ├── topicqa_stego.jsonl
│   ├── topicqa_cover_c1.jsonl
│   ├── topicqa_cover_c2.jsonl
│   ├── story_stego.jsonl
│   ├── story_cover_c1.jsonl
│   ├── story_cover_c2.jsonl
│   ├── litreview_stego.jsonl
│   ├── litreview_cover_c1.jsonl
│   └── litreview_cover_c2.jsonl
├── phase2_metrics/
│   ├── perplexity_all.jsonl
│   ├── token_counts.jsonl
│   └── stegoanalysis/
│       ├── classifier_2a_topicqa.json
│       ├── classifier_2a_story.json
│       ├── classifier_2b_topicqa.json
│       ├── classifier_2b_story.json
│       └── classifier_2b_litreview.json
├── phase3_attacks/
│   ├── topicqa_attacked.jsonl
│   ├── story_attacked.jsonl
│   └── litreview_attacked.jsonl
├── phase4_decode/
│   ├── topicqa_decoded.jsonl
│   ├── story_decoded.jsonl
│   └── litreview_decoded.jsonl
└── phase5_standalone/
    ├── ablations/
    ├── cross_model/
    ├── multi_turn/
    └── matched_payload/
```

------

## Phase 1: Text Generation (API-heavy)

**Goal:** Generate all text artifacts needed by every subsequent experiment.

### Shared setup

- **Prompts/seeds:** **300 prompts per system**, stored in `data/experiments/prompts/` — expansion complete. Existing prompt IDs are preserved; new entries appended with contiguous IDs. Provenance is recorded in each file's top-level `"expansion"` block. Expansion script: `experiments/utils/expand_prompts.py`. **Master seed: 42** (`numpy.default_rng`) for all RNG-based artifacts.
  - TopicQA: expanded 50 → 300 via a one-shot `gpt-4.1` call at temperature 0.9, JSON-mode, deduped case-insensitively against existing examples. New IDs 50–299. File: `data/experiments/prompts/topicqa_prompts.json`.
  - StorySlot: expanded 20 → 300 via two `gpt-4.1` calls at temperature 0.9, JSON-mode, deduped. New IDs 20–299. File: `data/experiments/prompts/story_prompts.json`.
  - LitReview: expanded 50 → 300 by sampling 250 additional paper IDs from the 1200-paper pool (`pca/litreview/references/papers.jsonl`) via `numpy.default_rng(42).permutation(1200)`, excluding the 50 already picked. File: `data/experiments/prompts/litreview_indices.json`.
- **Message bits:** Generated with `numpy.default_rng(42)` in a single draw sequence (order recorded in `messages.json` under `"draw_order"`: topicqa.stego, topicqa.c1, story.stego, story.c1, litreview.stego, litreview.c1). Per system: **300 stego bit strings + 300 C1 bit strings** at native capacity — TopicQA length 6, StorySlot length 18, LitReview length 20. Prompt index `i` (0–299) pairs 1:1 with message index `i`. File: `data/experiments/prompts/messages.json`.
- **ECC:** `RepetitionCode(1)` everywhere (no error correction — raw channel).
- **Encoder:** `BypassEncoder()` everywhere.

### Text types to generate

For each system, generate three types of text per prompt:

**Type S (stego):** `system.hide_message(message_bits, prompt)`. Record the generated text and all system state needed for later decoding.

Parameters:
model: response: gpt4.1; decoder: gpt4.1; temp: 0.7



**Type C1 (same-pipeline cover):** `system.hide_message(random_bits, prompt)` where `random_bits` is a *different* random bit string from the message. Goes through the identical pipeline — same subtopic generation, same response generation — but encodes different (random) bits. The classifier in Exp 2a should not be able to tell S from C1.

**Type C2 (prompted cover):** Call GPT-4.1 directly with a prompt that says:

- For TopicQA: `"Answer the following question in approximately {X} words.\n\nQuestion: {question}"` where X is the token count of the corresponding stego text (rounded to nearest 50).
- For StorySlot: `"Write a story of approximately {X} words based on the following premise.\n\nPremise: {premise}"` where X is matched to stego length.
- For LitReview: ```Write a realistic Related Work section for an academic paper on the topic of: "{paper_title} Cite as "LastName (YEAR)" for single authors or "LastName et al. (YEAR)" for multiple authors. Every provided reference must appear exactly only once. Write approximately {target_words} words. References you can use:
  {ref_list}``` 
- where reference_list is the full corpus for that paper, and the model picks freely.

**Important:** C2 must be generated AFTER the corresponding stego text, because the length target comes from the stego text's actual length.

### JSONL record schema

Each record in the phase1 output files must contain:

```json
{
  "id": "topicqa_s_042",
  "system": "topicqa",
  "text_type": "stego",
  "prompt": "What are the pros and cons of remote work?",
  "prompt_idx": 42,
  "message_bits": [0, 1, 1, 0, 0, 1],
  "text": "The generated text...",
  "token_count": 387,
  "word_count": 312,
  "char_count": 1843,
  "metadata": {
    "selected_subtopics": ["Work-life balance", "Communication challenges", ...],
    "all_subtopics": [...],
    "groups": [[...], ...],
    "chunks": [[0], [1], [1], [0], [0], [1]]
  },
  "system_state": {
    "question": "What are the pros and cons of remote work?",
    "error_encoded_length": 6
  },
  "timestamp": "2025-06-01T12:00:00Z"
}
```

For C2 texts, `message_bits` is null, `metadata` is null, and `system_state` is null. Include a `length_target` field instead.

### Checkpointing

Use append-mode JSONL writes so that if generation is interrupted, you can resume from the last written record. Before starting, scan existing output files to determine which (system, text_type, prompt_idx) combinations are already done.

### Sample sizes

Symmetric across all three systems: **300 prompts × 1 message × 3 text types = 900 texts per system**, totaling 2700 texts across systems.

- Per system: 300 S + 300 C1 + 300 C2
- One (prompt, message) pair → one stego text; prompt index `i` is paired with message index `i`.

**Why 300 per class**: Exp 2 stegoanalysis uses a transformer-based classifier (DistilBERT/RoBERTa fine-tuned). At N=300 per class, binomial SE is ~2.9%, giving a clean null result (~50% ± 3%). Fewer samples would under-power a fine-tuned transformer and risk overfit. The 300/300/300 split is uniform across systems so that cross-system comparisons aren't confounded by sample-size asymmetry.

**Reuse for Phase 3**: Phase 3 attacks consume only the **first 30 S stegos per system** (prompt indices 0–29). The remaining 270 S per system are used by Exp 2 (stegoanalysis) and Exp 5 (perplexity / quality). No extra stego generation for attacks; the attack-variance signal comes from 3 runs per stochastic attack (see Phase 3), not from more stegos.

This replaces the prior "2 stegos per prompt" (TopicQA) and "5 messages per prompt" (Story/LitReview) patterns — those were workarounds to stretch small samples across multiple experiments, and are no longer needed once Phase 1 is sized for transformer stegoanalysis and Phase 3 uses explicit attack runs.

------

## Phase 2: Local Metrics (Free — no API calls)

**Goal:** Compute all metrics that don't require API calls, using only the texts from Phase 1.

### 2a. Token/word counts and token efficiency

Already captured in Phase 1 records. Write a utility script `utils/token_counter.py` that:

- Counts tokens using `tiktoken` with the `o200k_base` encoding (GPT-4.1's tokenizer).
- Counts words (whitespace split).
- Computes bits_per_token = message_bit_count / token_count for stego texts.
- Aggregates per system: mean, std, min, max for token count, word count, bits_per_token.
- Outputs a summary JSON and a CSV suitable for plotting the token efficiency figure.

### 2b. Perplexity

Write `utils/perplexity.py` (or reuse existing one) that:

- Loads GPT-2 large from HuggingFace (`gpt2-large`).
- Computes per-token log-likelihood via `model.forward()`, then perplexity = exp(mean negative log-likelihood).
- Processes all texts from Phase 1 (all types, all systems).
- Outputs JSONL with fields: `id`, `system`, `text_type`, `perplexity`, `mean_nll`, `num_tokens`.
- Also compute per-system, per-type summary statistics (mean, std, median perplexity).

This feeds **Experiment 5** (text quality) and part of **Experiment 2** (stegoanalysis perplexity comparison).

### 2c. Stegoanalysis classifier

Write `experiments/exp2_stegoanalysis.py` that:

**Experiment 2a (stego vs same-pipeline cover):**

- For each system (TopicQA, StorySlot — NOT LitReview, since for metadata channels this is trivially at-chance):
  - Load all **300 stego texts (Type S)** and **300 same-pipeline cover texts (Type C1)**.
  - **Primary classifier**: fine-tune a transformer (DistilBERT or RoBERTa-base) with a binary classifier head. Use 5-fold stratified cross-validation (~240 train / 60 test per class per fold). Track train/val loss to confirm no overfit; use early stopping.
  - **Baseline classifier**: logistic regression on Qwen3-Embedding-4B embeddings, same 5-fold CV. Serves as a sanity check — under the null, the transformer should not substantially beat the baseline.
  - Report: accuracy, F1, AUC per fold, and mean ± std across folds, for both classifiers.
  - Also run Mann-Whitney U test comparing perplexity distributions of S vs C1.
- Target: ~50% accuracy (no distinguishable signal). At N=300 per class, binomial SE is ~2.9%, so 50% ± 3% is a clean null.

**Experiment 2b (stego vs prompted cover):**

- For each system (TopicQA, StorySlot, LitReview):
  - Load 300 S + 300 C2.
  - Same classifier setup as 2a (transformer primary + logistic baseline).
  - Report the same metrics.
- If accuracy is above chance, report feature importances (from the logistic baseline) and saliency/attention analysis (from the transformer) to identify what the classifier keys on.

**Additional stegoanalysis signals:**

- **LLM-as-judge**: few-shot prompt a separate model (e.g., Claude Sonnet 4.6 or GPT-4o) to classify stego vs cover. Report per-system accuracy on the same 300 + 300 split. Useful as a third independent measurement.
- **Perplexity-only classifier**: single-feature logistic regression on GPT-2 perplexity. Establishes whether perplexity alone is detectable.

**Output:** Per-system JSON files with fold-level results, confusion matrices, and summary statistics.

### 2d. Perplexity comparison (part of Exp 2)

In the same `exp2_stegoanalysis.py` or a companion script:

- For each system, compare perplexity distributions: S vs C1, S vs C2.
- Mann-Whitney U test, report U statistic and p-value.
- Generate box plots of perplexity by text type per system.

------

## Phase 3: Apply Attacks (API-heavy)

**Goal:** Apply all attack types to all stego texts. Each attacked text is saved once and reused for both Experiment 1 (attack validation metrics) and Experiment 3 (BER after decode).

### Attack configurations

**Design rationale**: Our three systems encode at the semantic-selection level (topic / slot / reference choice). Local attacks (synonym, sentence-level paraphrase, sentence-level back-translation) operate on surface-level features and are expected to leave semantic selections intact — we expect near-ceiling recovery at all local tampering levels. Local attacks therefore use a coarser grid (3 levels instead of 6) just to establish the flat ceiling curve. Global attacks are the discriminating attacks since they can rearrange and rewrite the semantic content that carries the message.

**Attack-variance sampling**: Each stochastic attack is run 3 times per stego to sample attack variance (paraphrase / translation outputs differ across runs at temperature > 0). The WordNet-based synonym attack is deterministic and needs only 1 run.

```python
attack_configs = [
    # Local attacks (coarser tampering grid, 3 levels)
    {"label": "synonym", "attack_type": "synonym", "local": True,
     "tampering_levels": [0.2, 0.5, 1.0], "runs_per_stego": 1},
    {"label": "local_paraphrase", "attack_type": "paraphrase", "local": True,
     "tampering_levels": [0.2, 0.5, 1.0], "runs_per_stego": 3},
    {"label": "local_backtranslation", "attack_type": "translate", "local": True,
     "tampering_levels": [0.2, 0.5, 1.0], "runs_per_stego": 3},

    # Global attacks (tampering = 1.0, local = False)
    {"label": "global_paraphrase", "attack_type": "paraphrase", "local": False,
     "tampering_levels": [1.0], "runs_per_stego": 3},
    {"label": "global_backtranslation", "attack_type": "translate", "local": False,
     "tampering_levels": [1.0], "runs_per_stego": 3},
]
```

**Attack count per system**: 30 stegos × (synonym: 3 levels × 1 run + local_paraphrase: 3 × 3 + local_BT: 3 × 3 + global_paraphrase: 1 × 3 + global_BT: 1 × 3) = 30 × 27 = **810 attacked texts per system**. Total across systems: **2430 attacked texts**.

### Procedure

For each system, use the **first 30 S stegos from Phase 1** (prompt indices 0–29). For each attack config, for each tampering level, for each run index (0 to `runs_per_stego - 1`):

1. Apply the attack: `attack(stego_text, tampering_level, local)`.
2. Save the attacked text with `run_idx` recorded.

Also apply the same attacks to **20 cover texts (Type C1, prompt indices 0–19)** per system for Experiment 1 attack validation metrics — we need original-vs-attacked pairs to compute BERTScore/BLEU/TER, and cover texts work fine for this (no stego involvement needed). Use 1 run per attack config for cover texts — attack variance on covers doesn't feed into any decoding step, so repeated runs aren't needed there.

### JSONL record schema

```json
{
  "id": "topicqa_s_000_global_paraphrase_1.0_run0",
  "source_id": "topicqa_s_000",
  "system": "topicqa",
  "attack_label": "global_paraphrase",
  "attack_type": "paraphrase",
  "local": false,
  "tampering_level": 1.0,
  "run_idx": 0,
  "original_text": "...",
  "attacked_text": "...",
  "original_token_count": 387,
  "attacked_token_count": 352,
  "system_state": { "question": "...", "error_encoded_length": 6 }
}
```

### Checkpointing

Same append-mode JSONL strategy. Before each attack call, check if (source_id, attack_label, tampering_level, run_idx) already exists in the output file.

------

## Phase 4: Decode and Measure (API for decode, free for metrics)

**Goal:** Decode all attacked stego texts to get BER, and compute attack severity metrics on all attacked texts for Experiment 1.

### 4a. Decode attacked stego texts

For each attacked stego text from Phase 3:

1. Restore system state from the record (`system_state` fields).
2. Call `system.recover_message(attacked_text)`.
3. Compare recovered bits to original `message_bits`.
4. Compute bitwise accuracy and perfect recovery (binary).

**Output JSONL:**

```json
{
  "source_id": "topicqa_s_000",
  "system": "topicqa",
  "attack_label": "global_paraphrase",
  "tampering_level": 1.0,
  "run_idx": 0,
  "original_bits": [0, 1, 1, 0, 0, 1],
  "recovered_bits": [0, 1, 0, 0, 0, 1],
  "bitwise_accuracy": 0.833,
  "bit_error_rate": 0.167,
  "perfect_recovery": false,
  "num_bit_errors": 1
}
```

One record per (source_id, attack_label, tampering_level, run_idx). For Phase 3's 810 attacks per system, Phase 4a produces 810 decode records per system.

### 4b. Attack severity metrics (Experiment 1)

For each attacked text (both stego and cover), compute:

- **BERTScore** F1 (original vs attacked) — `bert_score` pip package.
- **Cosine similarity** of paragraph embeddings (`all-mpnet-base-v2`).
- **BLEU** — `sacrebleu` package, original as reference, attacked as hypothesis.
- **TER** — `sacrebleu` or `pyter`.

Write `experiments/exp1_attack_validation.py` that:

- Reads Phase 3 attacked texts.
- Computes all four metrics per attacked text.
- Aggregates per (system, attack_label, tampering_level): mean ± std for each metric.
- Outputs summary tables as TSV and JSON.

This produces the attack validation table for the paper: for each attack type and tampering level, BERTScore should be high (semantic preservation) while BLEU should be low (surface disruption).

### 4c. Main results aggregation (Experiment 3)

Write `experiments/exp3_main_results.py` that:

- Reads Phase 4a decode results.
- Aggregates over the 3 runs per (source_id, attack_label, tampering_level) first (mean BER across runs per stego), then across the 30 stegos per (system, attack_label, tampering_level): mean BER, std BER, perfect recovery rate. Reporting std across stegos gives a between-stego error bar; the runs-per-stego averaging soaks up attack variance.
- Cross-references with token efficiency from Phase 2a.
- Outputs the main results table: system | capacity (bits) | bits/token | BER (no attack) | BER (global paraphrase) | BER (global BT) | perfect recovery rate.
- Also outputs parameterized attack curves: BER vs tampering_level per system per attack type, with error bars across stegos.

------

## Phase 5: Standalone Experiments

These require fresh API calls that can't be reused from Phases 1-4.

### 5a. Ablations (Experiment 4) — TopicQA only

Write `experiments/exp4_ablations.py`:

- **Group size ablation:** Run TopicQA with group_size ∈ {2, 4} on 20 prompts. For each, generate stego, apply global paraphrase, decode. Compare BER.
- **Subtopic count ablation:** Run TopicQA with n_subtopics ∈ {8, 12, 16, 20} on 20 prompts. Same procedure.
- Output: ablation results table (configuration | capacity | BER).
- Also run LitReview at 20 and 40 bits under global paraphrase, decode both, confirm BER is comparable.

### 5b. Cross-model robustness (Experiment 7)

Write `experiments/exp7_cross_model.py`:

- Reuse the **first 30 S stegos per system** from Phase 1 (same subset as Phase 3 — no regeneration needed).
- Apply global paraphrase using 3 different models: GPT-4o, Claude Sonnet 4.6 (via Anthropic SDK), Gemini (via Google SDK). 3 runs per stego per model to sample attack variance.
- Decode each and compute BER.
- Report per-model, not aggregated.
- For TopicQA primarily; StorySlot if budget allows.

### 5c. Multi-turn scaling (Experiment 6)

Write `experiments/exp6_multi_turn.py`:

- For TopicQA: run 3-5 turn conversations on 10 example question chains.
- Each turn encodes a fresh payload into a new response.
- Show: total bits = bits_per_turn × turns, BER stable across turns.
- Small scale, for discussion section only.

### 5d. Matched-payload experiment (Experiment 3b — apples-to-apples)

Write `experiments/exp3b_matched_payload.py`.

**Purpose**: Isolate encoding *method* from encoding *payload size*. At native capacity, LitReview hides 20 bits vs TopicQA's 6, so any BER difference in the main results confounds method and payload. Running all three systems at a shared 6-bit payload gives a clean method-level comparison under attack.

**Setup**:
- All three systems at **6 bits** of payload.
- 15 prompts per system × 1 message × 1 stego generation = 15 stegos per system, 45 total.
- Requires fresh generation for StorySlot (encode 6 bits — use first 6 slots, or random 6; document the choice) and LitReview (6-bit payload, fewer references selected). TopicQA at 6 bits is native and can reuse the first 15 Phase 1 stegos directly.
- Save fresh generations under `data/experiments/phase5_standalone/matched_payload/`.

**Attacks**: Only the discriminating attacks.
- `global_paraphrase` at tampering=1.0, 3 runs per stego.
- `global_backtranslation` at tampering=1.0, 3 runs per stego.
- 15 × 2 × 3 = **90 attacked texts per system, 270 total**.

**Output**: Table comparing mean BER at matched 6-bit payload across the three systems, plus error bars across stegos. Goes into the paper as a dedicated paragraph showing the method-only comparison. Contrast against the native-capacity main table (Phase 4c) to tell the full story: "at matched payload, methods are X; at native capacity, they trade off capacity for robustness as Y."

**Note on capacity treatment across experiments** (recap of decisions):
- Native capacity (6 / 18 / 20 bits): Phase 1 generation, Exp 2 stegoanalysis, Exp 3 main results, Exp 5 quality, Exp 7 cross-model, token-efficiency figure.
- Matched 6-bit payload: Exp 3b (this subsection).
- Varied within one system: Exp 4 ablations (TopicQA n_subtopics, LitReview 20 vs 40 bits).

------

## Utility Scripts

Create these utility modules under `utils/`:

### `utils/token_counter.py`

```python
def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    """Count tokens using tiktoken."""

def count_words(text: str) -> int:
    """Whitespace-split word count."""

def compute_bits_per_token(message_bits: list[int], token_count: int) -> float:
    """Bits per token efficiency metric."""

def length_summary(records: list[dict]) -> dict:
    """Aggregate token/word count statistics from Phase 1 records."""
```

### `utils/perplexity.py`

```python
def compute_perplexity(text: str, model, tokenizer, device="cuda") -> dict:
    """Compute perplexity using a HuggingFace causal LM.
    Returns {"perplexity": float, "mean_nll": float, "num_tokens": int}."""

def batch_perplexity(texts: list[str], model_name="gpt2-large", batch_size=8) -> list[dict]:
    """Batch perplexity computation."""
```

### `utils/metrics.py`

```python
def compute_bertscore(original: str, attacked: str) -> dict:
    """BERTScore F1 between original and attacked text."""

def compute_cosine_similarity(original: str, attacked: str, model_name="all-mpnet-base-v2") -> float:
    """Cosine similarity of sentence-transformer embeddings."""

def compute_bleu(reference: str, hypothesis: str) -> float:
    """BLEU score using sacrebleu."""

def compute_ter(reference: str, hypothesis: str) -> float:
    """TER using sacrebleu."""

def compute_bit_error_rate(original: list[int], recovered: list[int]) -> dict:
    """Returns {"ber": float, "bitwise_accuracy": float, "num_errors": int, "perfect": bool}."""
```

### `utils/io.py`

```python
def append_jsonl(path: str, record: dict):
    """Append a single JSON record to a JSONL file."""

def read_jsonl(path: str) -> list[dict]:
    """Read all records from a JSONL file."""

def get_completed_ids(path: str, key_fields: list[str]) -> set[tuple]:
    """Scan existing JSONL to find which combinations are already done.
    Returns set of tuples for resumability."""
```

### `utils/system_factory.py`

```python
def create_system(system_name: str, client, local_client, local_model, **overrides):
    """Factory to instantiate TopicQA, StorySlot, or LitReview with standard defaults."""

def restore_system_state(system, state_dict: dict):
    """Restore internal state (_question, _error_encoded_length, etc.) for decoding."""
```

------

## Figures to Generate

### Token efficiency figure

- Grouped bar chart: x-axis = system (TopicQA, StorySlot, LitReview), y-axis = bits per token.
- Include capacity (total bits) as text annotation on each bar.
- Use matplotlib, save as PDF for paper inclusion.
- Script: `figures/token_efficiency.py`, reads from Phase 2a output.

### Attack robustness curves (Experiment 3)

- Line plots: x-axis = tampering level, y-axis = BER (or bitwise accuracy).
- One plot per system, lines for each parameterized attack type.
- Separate figure for global attacks (bar chart: BER per system per global attack).
- Script: `figures/robustness_curves.py`, reads from Phase 4c output.

### Stegoanalysis results (Experiment 2)

- Bar chart: classifier accuracy per system, per variant (2a vs 2b), with 50% chance line.
- Box plots: perplexity distributions by text type per system.
- Script: `figures/stegoanalysis.py`, reads from Phase 2c/2d output.

------

## Implementation Notes

- Every script should be runnable standalone with `python -m experiments.phase1_generate` etc.
- Use `argparse` for all scripts so systems and sample sizes can be overridden.
- Print progress to stderr, data to files only.
- All API calls should have retry logic (3 attempts, exponential backoff).
- Set random seeds everywhere for reproducibility. Master seed: 42.
- For LitReview, load corpus once and pass it to the constructor rather than letting it load from disk on every instantiation.
- Token counting for C2 length matching: compute after generating each stego text, before generating the corresponding C2 text. This means within Phase 1, the generation order per prompt is: S first → measure length → C1 → C2 (with length target from S).
- For Phase 4a decoding, be careful to restore system state correctly. For TopicQA, set `system._question = state["question"]` and `system._error_encoded_length = state["error_encoded_length"]`. For LitReview, set `system._error_encoded_length = state["error_encoded_length"]`.