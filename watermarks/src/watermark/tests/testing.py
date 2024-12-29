from typing import List, Tuple
import torch
import pdb
from tqdm import tqdm
import os

# Use absolute imports since we're in a package
from watermark import (
    NanoGPTModel,
    GPT2Model,
    AESPRF,
    DeltaPerturb,
    Embedder,
    Extractor,
    set_seed
)
from watermark.utils import detect, get_z_score, get_limit, generate_n_grams_with_counts, count_maintained_n_grams

def find_delta(embedder: Embedder, extractor: Extractor, keys: List[bytes], h: List[str], m: List[int], c: int):
    """Test different delta values to find optimal strength."""
    deltas = [0.00001, 0.0001, .001, .01, .1, .5]
    delta_counters = [0 for _ in deltas]

    # Search for what delta works best
    for _ in range(10):
        for delta in deltas:
            # Generate watermarked text
            ct, tokens = embedder.embed(keys, h, m, delta, c, get_limit(len(m)))

            # Extract and count
            recovered_counters, _ = extractor.extract(keys, h, ct, c)
            result = recovered_counters[0]
            delta_counters[deltas.index(delta)] += result

    avg_delta_counters = [x / 10 for x in delta_counters]
    print("Average counters for each delta:", avg_delta_counters)

    detection = [get_z_score(get_limit(len(m)) * 10, x) for x in delta_counters]
    print("Z-scores for each delta:", detection)

def mismatch(enc_ids: List[int], dec_ids: List[int], c: int, tokenizer):
    """Analyze differences between encoded and decoded token sequences."""
    enc = [tokenizer.decode([t]) for t in enc_ids]
    dec = [tokenizer.decode([t]) for t in dec_ids]
    print("Encoded sequence:", enc)
    print("Decoded sequence:", dec)

    total_c_grams = sum(generate_n_grams_with_counts(enc, c).values())
    print('Total enc c-grams:', total_c_grams)
    maintained_c_grams = count_maintained_n_grams(enc, dec, c)
    print('Maintained c-grams:', maintained_c_grams)

def test(embedder: Embedder, extractor: Extractor, keys: List[bytes], h: List[str],
         m: List[int], delta: float, c: int, covertext_length: int) -> Tuple[List[int], dict, dict, bool]:
    """Test embedding and extraction."""
    ct, tokens = embedder.embed(keys, h, m, delta, c, covertext_length)
    recovered_counters, decode_tokens = extractor.extract(keys, h, ct, c)
    m_prime = [1 if detect(covertext_length, x) else 0 for x in recovered_counters]

    # Compare tokens
    enc_ids = tokens['input_ids'][0].tolist()
    dec_ids = decode_tokens['input_ids'][0].tolist()
    tokens_match = enc_ids == dec_ids

    return m_prime, tokens, decode_tokens, tokens_match

def test_with_logging(embedder: Embedder, extractor: Extractor, keys: List[bytes],
                     h: List[str], m: List[int], delta: float, c: int, covertext_length: int):
    """Test with detailed logging."""
    global token_match

    ct, tokens = embedder.embed(keys, h, m, delta, c, covertext_length)
    print('-------------------')
    recovered_counters, decode_tokens = extractor.extract(keys, h, ct, c)

    enc_ids = tokens['input_ids'][0].tolist()
    dec_ids = decode_tokens['input_ids'][0].tolist()

    same = enc_ids == dec_ids
    print('Tokens match:', same)
    if same:
        token_match += 1
    if not same:
        mismatch(enc_ids, dec_ids, c, embedder.tokenizer)

    print('-------------------')
    print('Generated text:', ct)
    print('Recovered counters:', recovered_counters)
    print('Z-scores:', [get_z_score(get_limit(len(m)), x) for x in recovered_counters])
    m_prime = [1 if detect(get_limit(len(m)), x) else 0 for x in recovered_counters]
    print('Original message:', m)
    print('Recovered message:', m_prime)

    return m_prime

def batch_test(embedder: Embedder, extractor: Extractor, message_length: int,
               num_tests: int, covertext_length: int = 100):
    """Run multiple tests and collect statistics."""
    h = ["To be or not to be, that is the question."]

    print('Batch test')
    print('Initial History:', ''.join(h))
    print('-' * 50)
    print('Covertext Size:', covertext_length)
    print('Message length:', message_length)
    print('Number of tests:', num_tests)
    print('-' * 50)

    message_matches = 0
    token_matches = 0

    for test_num in tqdm(range(num_tests)):
        print(f'Test {test_num + 1}/{num_tests}')
        m = [torch.randint(2, (1,)).item() for _ in range(message_length)]
        keys = [os.urandom(32) for _ in range(message_length)]
        c = 3
        delta = 0.1

        m_prime, tokens, decode_tokens, tokens_match = test(
            embedder, extractor, keys, h, m, delta, c, covertext_length
        )

        message_match = m == m_prime
        if message_match:
            message_matches += 1
        if tokens_match:
            token_matches += 1

        if not message_match or not tokens_match:
            print('\nGenerated text:')
            enc_ids = tokens['input_ids'][0].tolist()
            dec_ids = decode_tokens['input_ids'][0].tolist()
            print('Encode:', ''.join([embedder.tokenizer.decode([t]) for t in enc_ids]))
            print('Decode:', ''.join([embedder.tokenizer.decode([t]) for t in dec_ids]))
            mismatch(enc_ids, dec_ids, c, embedder.tokenizer)

        print('\nResults:')
        print('m:      ', m)
        print('m_prime:', m_prime)
        print(f'Message Match: {message_match}')
        print(f'Tokens Match: {tokens_match}')
        print('-' * 50)

    print('\nFinal Results:')
    print(f'Message Matches: {message_matches}/{num_tests} ({message_matches/num_tests*100:.1f}%)')
    print(f'Token Matches: {token_matches}/{num_tests} ({token_matches/num_tests*100:.1f}%)')

if __name__ == "__main__":
    # Set seed for deterministic behavior
    # set_seed(9)
    
    # Initialize components
    model = NanoGPTModel()
    # model = GPT2Model()
    prf = AESPRF(vocab_size=model.vocab_size, max_token_id=model.vocab_size-1)
    perturb = DeltaPerturb()
    embedder = Embedder(model, model.tokenizer, prf, perturb)
    extractor = Extractor(model, model.tokenizer, prf)

    # Test parameters
    m = [1]  # Message to embed
    keys = [b'\0' * 32]  # PRF keys
    h = ['Hey! Want to get coffee?', 'Would Friday work for you?']  # History
    c = 5  # Context size
    delta = 0.001  # Perturbation strength
    covertext_length = 1500  # Length of generated text

    # Run tests
    # pdb.set_trace()
    batch_test(embedder, extractor, len(m), 30, covertext_length)
