# Robust Steganography

## Overview

`robust_steganography` is a project that encompasses two distinct systems for hiding messages within text: an embeddings-based steganography system and a watermarking-based system.

### Embeddings-Based System

The embeddings-based system leverages various components such as encoders, error correction codes, and hash functions to ensure robust message embedding and retrieval. It is designed to hide messages within text using advanced embedding techniques.

#### Enron Email Dataset Processing

The `enron_emails` directory contains utilities for processing the Sent Enron email dataset (http://www.verbs.colorado.edu/enronsent) to train the PCA hash model:

- **Data Cleaning**: Scripts to clean and preprocess the raw Enron email corpus
- **Embedding Generation**: Code to generate embeddings for both original emails and their paraphrases
- **Paraphrase Generation**: Tools to create semantically equivalent versions of emails using LLMs
- **Data Formatting**: Utilities to format the original and paraphrase embeddings for PCA model training

This processed data is used to train a PCA-based hash function that is robust to semantic-preserving transformations, which is a key component of the steganography system. To create other covertext modes, follow this example process and create a corresponding PCA Hash. Then add an appropriate system prompt to generate the covertext.

### Watermarking-Based System

The watermarking-based system modifies language model output distributions to embed watermarks in generated text. It supports both character-level (NanoGPT) and BPE-based (GPT2) models, offering different trade-offs between watermark reliability and text naturalness.

## Directory Structure

The project is organized as follows:

- **embeddings/**: Contains core components and utilities for the embeddings-based steganography system.
  - **src/core/**: Core components like encoders, error correction, hash functions, and the main steganography system.
  - **examples/**: Example scripts demonstrating how to use the embeddings-based steganography system.
  - **temp_pca/**: Temporary files related to PCA model training and testing.
  - **src/utils/**: Utility functions for embedding and text processing.
  - **tests/**: Unit tests for various components of the embeddings-based system.
  - **setup.py**: Configuration for packaging and installing the embeddings-based system.

- **watermarks/**: Contains the watermarking-based system.
  - **src/watermark/**: Core watermarking implementation
    - **models/**: Language model implementations (NanoGPT, GPT2)
    - **core/**: Core embedder and extractor
    - **prf/**: Pseudorandom functions
    - **perturb/**: Distribution perturbation methods
    - **attacks/**: Attack implementations
    - **utils/**: Utility functions
    - **tests/**: Test suite
  - **examples/**: Example scripts demonstrating watermarking
  - **setup.py**: Package configuration

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Installation

### Embeddings System

To install the embeddings-based system in development mode, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NeilAPerry/robust_steganography.git
   cd robust_steganography
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Navigate to the embeddings directory and install the package in development mode**:
   ```bash
   cd embeddings
   pip install -e .
   ```

4. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up the environment variables**:
   - Copy the `.env.example` file to a new file named `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file to include your LLM key and any other necessary configurations.
   - Source the `.env` file to set the environment variables:
     ```bash
     source .env
     ```

### Watermarking System

1. **From the project root**:
   ```bash
   cd watermarks
   pip install -e .
   ```

## Usage

### Embeddings System

To run the example scripts for the embeddings-based system, navigate to the `embeddings/examples/` directory and execute the desired script. For instance:

```bash
python example.py
```

### Watermarking System

The watermarking system supports two language models:
1. **ShakespeareNanoGPT**: An example character-level model trained on Shakespeare's works. While character-level models are less common, they offer more reliable watermarking due to perfect token encoding/decoding (each character maps to exactly one token). Users can train their own character-level models on any text distribution and use them with this library - Shakespeare is just one example to demonstrate the benefits of character-level tokenization.
2. **GPT2**: More natural text generation but less reliable watermarking due to BPE tokenization

Basic example:
```python
from watermark import (
    ShakespeareNanoGPTModel,  # or GPT2Model
    AESPRF,
    SmoothPerturb,
    Embedder,
    Extractor
)

# Initialize components
model = ShakespeareNanoGPTModel()
prf = AESPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
perturb = SmoothPerturb()
embedder = Embedder(model, model.tokenizer, prf, perturb)
extractor = Extractor(model, model.tokenizer, prf)

# Embed watermark
message = [1, 0, 1]  # Message to hide
keys = [b'\x00' * 32, b'\x01' * 32, b'\x02' * 32]  # One key per bit
history = ["Initial context"]
watermarked_text, _ = embedder.embed(
    keys=keys,
    h=history,
    m=message,
    delta=0.1,
    c=5,
    covertext_length=100
)

# Extract watermark
recovered_counters, _ = extractor.extract(keys, history, watermarked_text, c=5)
```

#### Custom Components

The system is designed to be extensible. You can create custom models, PRFs, or perturbation methods by inheriting from the provided base classes:

```python
from watermark import LanguageModel, BaseTokenizer, PRF, PerturbFunction

# Custom language model
class MyCustomModel(LanguageModel):
    def __init__(self):
        self._tokenizer = MyCustomTokenizer()
        # Initialize your model...
    
    def get_next_token_distribution(self, input_tokens):
        # Implement your model's token prediction...
        pass

# Custom tokenizer
class MyCustomTokenizer(BaseTokenizer):
    def encode(self, text, return_tensors=None):
        # Implement tokenization...
        pass
    
    def decode(self, token_ids):
        # Implement detokenization...
        pass

# Custom PRF
class MyCustomPRF(PRF):
    def __call__(self, key: bytes, text: str, c: int) -> list:
        # Implement your PRF logic...
        pass

# Custom perturbation method
class MyCustomPerturb(PerturbFunction):
    def __call__(self, p, r, delta):
        # Implement your perturbation method...
        pass

# Use custom components
model = MyCustomModel()
prf = MyCustomPRF()
perturb = MyCustomPerturb()
embedder = Embedder(model, model.tokenizer, prf, perturb)
```

Additionally, the `covertext_calculator` module provides a way to calculate the minimum required covertext length for a given message length, error rate, and watermark strength.

```python
from watermark.covertext.smooth_calculator import SmoothCovertextCalculator

n_bits = 3  # Length of message to hide
epsilon = 0.05  # For 95% success probability
delta = 0.2  # Perturbation strength
safety_factor = 10

calculator = SmoothCovertextCalculator()
required_length = calculator.get_covertext_length(
    n=n_bits,
    epsilon=epsilon,
    delta=delta,
    safety_factor=safety_factor
)

# use this length as the covertext_length parameter in the embedder
```

For complete examples, see:
- `examples/shakespeare_nanogpt_example.py`: Character-level watermarking
- `examples/gpt2_example.py`: BPE-based watermarking
- `examples/harsh_perturb_example.py`: Alternative perturbation method
- `examples/covertext_calculator_example.py`: Calculating required covertext length example

### Attacks

The project includes several text modification attacks to test watermark robustness:

#### Threat Model Attacks
The watermarking scheme is designed to be robust against a specific family of attacks based on local text modifications:

1. **Synonym Substitution** (`attacks/synonym.py`)
   - Replaces words with their synonyms while preserving meaning
   - Uses WordNet to find valid substitutions
   - Configurable probability of replacing each eligible word
   - Preserves sentence structure and formatting

2. **N-gram Shuffling** (`attacks/ngram_shuffle.py`)
   - Breaks text into token n-grams and randomly shuffles them
   - Can operate globally or preserve sentence boundaries
   - Configurable n-gram size and shuffle probability
   - Handles both character-level and BPE tokenization

These attacks can be composed to form a "paraphrase-preserving Levenshtein edit distance" attack-- i.e. complex attacks within the paper's threat model can be constructed by combining synonym substitution and n-gram shuffling.

Example usage:
```python
from watermark.attacks import NGramShuffleAttack, SynonymAttack

# N-gram shuffle attack
attack = NGramShuffleAttack(
    model=model,
    n=3,  # n-gram size
    probability=0.5,  # probability of shuffling each n-gram
    local=True  # preserve sentence boundaries
)
modified_text = attack(original_text)

# Synonym substitution attack
attack = SynonymAttack(model=model, probability=0.3)
modified_text = attack(original_text)
```

#### Beyond the Threat Model

The project also includes a more powerful attack that falls outside the watermarking scheme's threat model:

**LLM-based Paraphrasing** (`attacks/paraphrase.py`)
- Uses GPT-4 to completely rephrase text while preserving meaning
- Can operate globally or sentence-by-sentence
- Configurable temperature for controlling variation
- Much stronger than local edit-based attacks
- Expected to defeat the watermarking scheme

While this attack is too strong for the watermarking system, the embeddings-based steganography system is specifically designed to be robust against such semantic-preserving transformations.

Example usage:
```python
from watermark.attacks import ParaphraseAttack
from openai import OpenAI

client = OpenAI()
attack = ParaphraseAttack(
    client=client,
    model="gpt-4o-mini",
    temperature=0.0,
    local=True  # paraphrase sentence-by-sentence
)
modified_text = attack(original_text)
```

For complete examples of each attack, see:
- `examples/ngram_shuffle_example.py`
- `examples/synonym_attack_example.py`
- `examples/paraphrase_example.py`

## Testing

### Embeddings System
Follow testing instructions in embeddings/README.md

### Watermarking System
```bash
cd watermarks
python -m pytest src/watermark/tests/
```

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - See LICENSE file for details
