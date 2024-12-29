# Robust Steganography

## Overview

`robust_steganography` is a project that encompasses two distinct systems for hiding messages within text: an embeddings-based steganography system and a watermarking-based system.

### Embeddings-Based System

The embeddings-based system leverages various components such as encoders, error correction codes, and hash functions to ensure robust message embedding and retrieval. It is designed to hide messages within text using advanced embedding techniques.

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
1. **NanoGPT (Character-level)**: More reliable watermarking due to character-by-character tokenization
2. **GPT2**: More natural text generation but less reliable watermarking due to BPE tokenization

Basic example:
```python
from watermark import (
    NanoGPTModel,  # or GPT2Model
    AESPRF,
    DeltaPerturb,
    Embedder,
    Extractor
)

# Initialize components
model = NanoGPTModel()
prf = AESPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
perturb = DeltaPerturb()
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

For complete examples, see:
- `examples/shakespeare_nanogpt_example.py`: Character-level watermarking
- `examples/gpt2_example.py`: BPE-based watermarking

## Testing

### Embeddings System
Follow testing instructions in embeddings/README.md

### Watermarking System
```bash
cd watermarks
python -m pytest src/watermark/tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - See LICENSE file for details
