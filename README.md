# Robust Semantic Steganography with Large Language Models

## Overview

This repo is the official codebase for the paper _Robust Semantic Steganography with Large Language Models_.
Inspired by [Perry et al. (2025)'s study](https://arxiv.org/abs/2504.08977) on robust steganography schemes,
we present a general steganography scheme that is secure and robust to extreme global paraphrase attacks using the semantic channels of various Natural Language Generation tasks. 

### Dynamic Semantic Unit Encoding Pipeline


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
- **data/experiments/**: Designated path for experiment data storage.
- **data/litreview/**: Reference corpus for the LitReview system, scraped from the Semantic Scholar API, shipped as a runnable example dataset. `references/` holds the corpus (`corpus.jsonl`, `references.jsonl`, `papers.jsonl`) and `semantic_scholar_requests.py` is the scraper used to regenerate it.
- **measurements/ (legacy)**: Complete scripts for the evaluation on Perry et al.'s steganography scheme.
- **pca/ (legacy)**: Complete scripts for generating datasets for training PCA models used by the legacy embedding-based models. (The LitReview reference corpus that used to live here has moved to `data/litreview/`.)


## Reproduction


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
