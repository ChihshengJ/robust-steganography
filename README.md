# Robust Semantic Steganography with Large Language Models

## Overview

This repo is the official codebase for the paper _Robust Semantic Steganography with Large Language Models_.
Inspired by [Perry et al. (2025)'s study](https://arxiv.org/abs/2504.08977) on robust steganography schemes,
we present a general steganography scheme that is secure and robust to extreme semantics-preserving attacks using the semantic channels of various Natural Language Generation tasks.

## Dynamic Semantic Unit Encoding Pipeline

This pipeline is aimed to solve two problems:

1. Given that natural language (or text) is a tight channel with relatively low entropy, how do we encode bits in this channel while making it invisible?
2. LLMs makes attacks on text steganography extremely cheap and effective, how do we utilize them to encode secrets that withstand the extremest attacks?

We propose this dynamic semantic unit encoding pipeline that partially solves these two problems by encoding secret bits within semantic channels.

<p align="center">
  <img src="https://raw.githubusercontent.com/ChihshengJ/robust-steganography/refs/heads/main/resources/figure_1.png" width="60%">
</p>

For any monitored channel to still be functional, attacks on alleged-steganography content needs to be at least semantics-preserving.
This means that global paraphrase attack (or other similar attacks) is the attack that poses the greatest threats to steganographic communication.
The frivolous nature of generated contents opens up the possibility to encode secret bits in the semantic space.

### Legacy Embedding-based systems

The legacy embedding-based systems are extensions of Perry et al.'s embedding steganography scheme.
We switched the sentence-based system with a story plotline-based system, a event summary-based system, and a unit test system.
The general idea is that instead using sentences as the base for bit-encoding and rejection sampling, we use more abstract semantic units that can be retrieved by LLMs to encode the payload.
This design not only solved the bit ordering issue that global paraphrase attacks might introduce, but also ensures the

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

The experiments run as four phases. Bash wrappers in [`scripts/`](scripts/) chain them;
the underlying modules (`python -m experiments.phase*`) are fully parameterized if you want finer control.

### Install

```bash
uv sync                       # installs the package + deps from uv.lock
cp .env.example .env          # then fill in OPENAI_API_KEY (see the file)
```

`scripts/*` auto-detect uv: if `uv` is on your PATH they run through
`uv run python` (the project's `.venv`, no activation needed), otherwise they
fall back to plain `python`. To force a specific interpreter, export `PYTHON`
(e.g. `PYTHON="uv run python"` or `PYTHON=python3.12`).

Extra one-time data some steps need:

- **NLTK** (synonym attack / style diagnostics): `python -m nltk.downloader wordnet punkt vader_lexicon`
- **HuggingFace models** download automatically on first use (`gpt2-large`, `distilbert-base-uncased`, `all-mpnet-base-v2`, BERTScore).

### What each phase needs

| Phase                      | Script               |    OpenAI API     |       llama.cpp server       | GPU (else slow CPU/MPS) |
| -------------------------- | -------------------- | :---------------: | :--------------------------: | :---------------------: |
| 1. Generate texts          | `phase1_generate.sh` |     required      |  Long-form QA/StoryGen only  |            —            |
| 2. Metrics + stegoanalysis | `phase2_metrics.sh`  |     optional¹     |              —               |       recommended       |
| 3. Apply attacks           | `phase3_attacks.sh`  |     required      |              —               |            —            |
| 4. Decode + score          | `phase4_decode.sh`   | required (decode) | Long-form QA/StoryGen decode |    recommended (4b)     |

¹ Only the optional `RUN_LLM_JUDGE` / `RUN_EMBEDDINGS` stegoanalysis signals
call out (OpenRouter / Google); the default Phase 2 signals are local-only.

### Local model server (Long-form QA & StoryGen)

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
scripts/smoke_test.sh   # imports + no-API dry-runs; no OpenAI credit spent
```

### Full run

```bash
scripts/run_all.sh  # both tracks at native capacity (API + GPU heavy)
scripts/run_all.sh detectability    # or run one track
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

### Experiment Scripts to Paper Mapping

| Paper experiment                                             | Produced by                                                                 |
| ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Main recovery results / attack curves                        | `phase4_decode/phase4c_main_results.py`                                     |
| Steganalysis (classifier, embeddings, LLM-judge, perplexity) | `phase2_metrics/phase2c_*` + `phase1_generation`                            |
| Attack validation (BERTScore/BLEU/TER)                       | `phase4_decode/phase4b_attack_metrics.py`                                   |
| Text quality (token efficiency, perplexity)                  | `phase2_metrics/phase2a_token_counts.py`, `phase2b_perplexity.py`           |
| Style/genre diagnostics                                      | `phase2_metrics/phase2d_genre_diagnostic.py`, `phase2c_ngram_diagnostic.py` |

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
   - Uses GPT-4.1 to completely rephrase text while preserving meaning
   - Can operate globally or sentence-by-sentence
   - Configurable temperature for controlling variation
   - Much stronger than local edit-based attacks

4. **Round-Trip Translation** (`attacks/translation.py`)
   - Uses GPT-4.1 to translate stegtexts from Language A to language B, then back to language A
   - Can operate globally or sentence-by-sentence
   - Configurable temperature for controlling variation

## Use Cases

The steganography systems in this library can be used for various privacy-preserving and information-hiding applications:

### Censorship Resistance

- Enable secure communication in environments with active censorship or surveillance
- Protect messages from being altered or tampered with during transmission
- Allow verification of message authenticity even if intermediaries modify the text

### For AI Agents

- Secret communication that survives communication sessions
- Evade thought-monitoring from humans

Note: This tool is intended for legitimate privacy-preserving use cases. Users are responsible for complying with all applicable laws and regulations in their jurisdiction.

## License

MIT License - See LICENSE file for details
