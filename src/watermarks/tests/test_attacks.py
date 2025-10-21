"""Tests for watermarking attacks."""
import json
import base64
from typing import List, Dict, Any
from pathlib import Path

from ..attacks.synonym import SynonymAttack
from ..core.extractor import Extractor
from ..utils.helpers import (
    detect, get_limit, generate_n_grams_with_counts, 
    count_maintained_n_grams
)

def load_examples(data_path: Path) -> List[Dict[str, Any]]:
    """Load test examples from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    # Convert keys from base64 to bytes
    for example in data:
        example['keys'] = [base64.b64decode(key) for key in example['keys']]
    return data

def test_synonym_attack(
    examples: List[Dict[str, Any]], 
    extractor: Extractor,
    log_file: Path
):
    """Test synonym replacement attack against watermarks."""
    attack = SynonymAttack()
    survived = []

    with open(log_file, 'a') as log:
        for example in examples:
            # Apply attack
            original = example['ct']
            modified = attack(original)
            
            # Try to recover message
            keys = example['keys']
            h = example['h']
            c = example['c']
            recovered_counters, decode_tokens = extractor.extract(
                keys, h, modified, c
            )
            m_prime = [
                1 if detect(get_limit(None), x) else 0 
                for x in recovered_counters
            ]

            # Analyze n-gram preservation
            original_tokens = extractor.tokenizer.encode(original)
            modified_tokens = extractor.tokenizer.encode(modified)
            
            original_n_grams = sum(
                generate_n_grams_with_counts(original_tokens, c).values()
            )
            n_grams_common = count_maintained_n_grams(
                original_tokens, modified_tokens, c
            )

            # Log results
            log_attack_result(
                log, original, modified, example, 
                original_n_grams, n_grams_common, m_prime
            )

            if m_prime == example['m']:
                survived.append((original, modified))

    print(f'Survived attacks: {len(survived)}/{len(examples)}')
    return survived

def log_attack_result(f, original, modified, example, orig_grams, common_grams, m_prime):
    """Log detailed attack results."""
    f.write(f'Original: {original}\n')
    f.write(f'Modified: {modified}\n')
    f.write(f'Keys: {example["keys"]}\n')
    f.write(f'History: {example["h"]}\n')
    f.write(f'Message: {example["m"]}\n')
    f.write(f'Context size: {example["c"]}\n')
    f.write(f'Original n-grams: {orig_grams}\n')
    f.write(f'Common n-grams: {common_grams}\n')
    f.write(f'N-gram preservation: {common_grams/orig_grams:.2%}\n')
    f.write(f'Recovered message: {m_prime}\n')
    f.write('-' * 50 + '\n') 