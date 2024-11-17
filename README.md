# Robust Steganography

## Overview

`robust_steganography` is a project that encompasses two distinct systems for hiding messages within text: an embeddings-based steganography system and a watermarking-based system. This README focuses on the embeddings-based system, which has been packaged for development use.

### Embeddings-Based System

The embeddings-based system leverages various components such as encoders, error correction codes, and hash functions to ensure robust message embedding and retrieval. It is designed to hide messages within text using advanced embedding techniques.

### Watermarking-Based System

The watermarking-based system is a separate component that focuses on embedding messages using watermarking techniques. This README does not cover its usage or installation.

## Directory Structure

The project is organized as follows:

- **embeddings/**: Contains core components and utilities for the embeddings-based steganography system.
  - **src/core/**: Core components like encoders, error correction, hash functions, and the main steganography system.
  - **examples/**: Example scripts demonstrating how to use the embeddings-based steganography system.
  - **temp_pca/**: Temporary files related to PCA model training and testing.
  - **src/utils/**: Utility functions for embedding and text processing.
  - **tests/**: Unit tests for various components of the embeddings-based system.
  - **setup.py**: Configuration for packaging and installing the embeddings-based system.

- **watermarks/**: Contains scripts and logs related to the watermarking system and its attacks.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Installation

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

## Usage

To run the example scripts for the embeddings-based system, navigate to the `embeddings/examples/` directory and execute the desired script. For instance:

```bash
python example.py
```
