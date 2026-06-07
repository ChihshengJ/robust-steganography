# Robust Semantic Steganography with Large Language Models

## Overview

This repo is the official codebase for the paper _Robust Semantic Steganography with Large Language Models_.
Inspired by [Perry et al. (2025)'s study](https://arxiv.org/abs/2504.08977) on robust steganography schemes,
we present a general steganography scheme that is secure and robust to extreme semantics-preserving attacks using the semantic channels of various Natural Language Generation tasks.

### Dynamic Semantic Unit Encoding Pipeline

This pipeline is aimed to solve two problems:

1. Given that natural language (or text) is a tight channel with relatively low entropy, how do we encode bits in this channel while making it invisible?
2. LLMs makes attacks on text steganography extremely cheap and effective, how do we utilize them to encode secrets that withstand the extremest attacks?

We propose this dynamic semantic unit encoding pipeline that partially solves these two problems by encoding secret bits within semantic channels.
![A simple illustration of how the pipeline works](https://github.com/ChihshengJ/robust-steganography/reasources/figure_1.png)

### Legacy Embedding-based models

## Directory Structure

The project is organized as follows:

- **src/**: Root of the project package, which includes the code for the steganography pipelines and attacks.
  - **systems/**: Root of the pipeline modules.
    - **core/**: Base class for all steganography pipelines and three pipelines mentioned in the paper.
      - **embeddings/ (legacy)**: Pipelines that use semantic hash and rejection sampling, which failed to be robust, hense dropped from the paper entirely.
    - **utils/**: utils for all steganography schemes, including text generation, rejection sampling, and backtracking.
    - **configs/**: Prompts and default settings for pipelines and the backtracker.
    - **path**: Routes PCA model artifacts to the embedding-based systems.
  - **attacks/**: Root of the attack modules, includes attack base class, ngram shuffle attack, paraphrase attack, synonym attack, and round-trip translation attack.
- **experiments/**: Contains scripts for dataset generation and all the experiments we conducted for the paper.
  - **dry_run/**: Small scale runs we used for exploration.
  - **phase1_generation/**: Scripts for generate the entire dataset for all experiments mentioned in the paper.
  - **phase2_metrics/**: Scripts for steganalysis experiments dataset generation and evaluation.
  - **phase3_attacks**: Script for generating attacked stegotexts.
  - **phase4_decode**: Scripts for decoding attacked stegotexts and generating final results for the recovery accuracy tests.
- **scripts/**: Bash wrappers that chain the phases for reproduction (`run_all.sh`, `smoke_test.sh`, per-phase scripts). See [`scripts/README.md`](scripts/README.md).
- **data/experiments/**: Designated path for experiment data storage.
- **data/litreview/**: Reference corpus for the LitReview system, scraped from the Semantic Scholar API, shipped as a runnable example dataset. `references/` holds the corpus (`corpus.jsonl`, `references.jsonl`, `papers.jsonl`) and `semantic_scholar_requests.py` is the scraper used to regenerate it.
- **measurements/ (legacy)**: Complete scripts for the evaluation on Perry et al.'s steganography scheme.
- **pca/ (legacy)**: Complete scripts for generating datasets for training PCA models used by the legacy embedding-based models. (The LitReview reference corpus that used to live here has moved to `data/litreview/`.)

## Reproduction

The experiments run as four phases. Bash wrappers in [`scripts/`](scripts/)
chain them; the underlying modules (`python -m experiments.phase*`) are fully
parameterized if you want finer control. `experiment.md` is the original design
doc and is partly outdated — **the code and this section are the source of
truth** (see the note at the top of that file).

### Install

```bash
uv sync                       # installs the package + deps from uv.lock
cp .env.example .env          # then fill in OPENAI_API_KEY (see the file)
```

`scripts/*` default to plain `python`; to run them under uv, export
`PYTHON="uv run python"`.

Extra one-time data some steps need:

- **NLTK** (synonym attack / style diagnostics): `python -m nltk.downloader wordnet punkt vader_lexicon`
- **HuggingFace models** download automatically on first use (`gpt2-large`, `distilbert-base-uncased`, `all-mpnet-base-v2`, BERTScore).

### What each phase needs

| Phase                      | Script               | OpenAI API  | Local llama.cpp server | GPU (else slow CPU/MPS) |
| -------------------------- | -------------------- | :---------: | :--------------------: | :---------------------: |
| 1. Generate texts          | `phase1_generate.sh` |     ✅      |   TopicQA/Story only   |            —            |
| 2. Metrics + stegoanalysis | `phase2_metrics.sh`  |  optional¹  |           —            |       recommended       |
| 3. Apply attacks           | `phase3_attacks.sh`  |     ✅      |           —            |            —            |
| 4. Decode + score          | `phase4_decode.sh`   | ✅ (decode) |  TopicQA/Story decode  |    recommended (4b)     |

¹ Only the optional `RUN_LLM_JUDGE` / `RUN_EMBEDDINGS` stegoanalysis signals
call out (OpenRouter / Google); the default Phase 2 signals are local-only.

### Local model server (TopicQA & Story)

TopicQA and Story decode by **re-generating** their subtopics/slots with a local
model, so generation and decoding must be byte-for-byte identical. Launch the
pinned llama.cpp server in a separate terminal and leave it up for Phases 1 & 4:

```bash
scripts/serve_local_model.sh /path/to/Qwen3.5-4B-UD-Q8_K_XL.gguf
```

Point elsewhere or swap the GGUF via `LOCAL_BASE_URL` / `LOCAL_MODEL` (see
`.env.example`). LitReview and the baseline don't use it. You can verify a serve
command reproduces a prior run with
`python -m experiments.check_subtopic_repro`.

### Quick check first (free)

```bash
scripts/smoke_test.sh          # imports + no-API dry-runs; no OpenAI credit spent
```

### Full run

```bash
scripts/run_all.sh             # both tracks at native capacity (API + GPU heavy)
scripts/run_all.sh detectability   # or run one track
scripts/run_all.sh robustness
```

The pipeline has two data tracks; `run_all.sh` does both:

- **Detectability** (steganalysis + quality): the full 300-per-class dataset at
  native capacity in top-level `phase1_texts/`, scored by Phase 2.
- **Robustness** (recovery under attack): the first 30 stegos per system in
  `{system}_cap{N}/`, attacked in Phase 3 and decoded in Phase 4. Native
  capacities are TopicQA 6, Story 18, LitReview 20, baseline 3 bits.

Running phases by hand (everything is checkpoint-resumable; re-running skips
finished records):

```bash
# Detectability
SYSTEM=topicqa SUBDIR='' scripts/phase1_generate.sh
scripts/phase2_metrics.sh

# Robustness (per system, with its native capacity)
SYSTEM=litreview CAPACITY=20 scripts/phase1_generate.sh
SYSTEM=litreview CAPACITY=20 scripts/phase3_attacks.sh
SYSTEM=litreview CAPACITY=20 scripts/phase4_decode.sh
```

See [`scripts/README.md`](scripts/README.md) for every env knob.

### Where experiments live (phase → paper)

| Paper experiment                                                     | Produced by                                                                 |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Exp 1 — attack validation (BERTScore/BLEU/TER)                       | `phase4_decode/phase4b_attack_metrics.py`                                   |
| Exp 2 — steganalysis (classifier, embeddings, LLM-judge, perplexity) | `phase2_metrics/phase2c_*` + `phase1_generation/phase1a_launder.py`         |
| Exp 3 — main recovery results / attack curves                        | `phase4_decode/phase4c_main_results.py`                                     |
| Exp 5 — text quality (token efficiency, perplexity)                  | `phase2_metrics/phase2a_token_counts.py`, `phase2b_perplexity.py`           |
| Style/genre diagnostics                                              | `phase2_metrics/phase2d_genre_diagnostic.py`, `phase2c_ngram_diagnostic.py` |

### Attacks

The project includes several text modification attacks to test robustness:

#### Threat Model Attacks

1. **N-gram Shuffling** (`attacks/ngram_shuffle.py`)
   - Breaks text into token n-grams and randomly shuffles them
   - Can operate globally or preserve sentence boundaries
   - Configurable n-gram size and shuffle probability
   - Handles both character-level and BPE tokenization

2. **Synonym Substitution** (`attacks/synonym.py`)
   - Replaces words with their synonyms while preserving meaning
   - Uses WordNet to find valid substitutions
   - Configurable probability of replacing each eligible word
   - Preserves sentence structure and formatting

3. **LLM-based Paraphrasing** (`attacks/paraphrase.py`)

- Uses GPT-4 to completely rephrase text while preserving meaning
- Can operate globally or sentence-by-sentence
- Configurable temperature for controlling variation
- Much stronger than local edit-based attacks
- Expected to defeat the watermarking scheme

## Use Cases

The steganography systems in this library can be used for various privacy-preserving and information-hiding applications:

### Censorship Resistance

- Enable secure communication in environments with active censorship or surveillance
- Protect messages from being altered or tampered with during transmission
- Allow verification of message authenticity even if intermediaries modify the text

### Covert File Storage

- Hide entire files within seemingly innocent text documents
- Convert binary data into natural-looking text for border crossings or inspections
- Store sensitive information in plain sight

### Cloud Storage Privacy

- Store private files on cloud platforms disguised as creative writing
- Make sensitive data appear as:
  - Collections of poetry or short stories
  - Novel drafts or writing exercises
  - Personal journal entries
  - Blog posts or articles
- Avoid drawing attention to encrypted files while maintaining data privacy

Note: This tool is intended for legitimate privacy-preserving use cases. Users are responsible for complying with all applicable laws and regulations in their jurisdiction.

## License

MIT License - See LICENSE file for details
