import logging
import concurrent.futures
import numpy as np
from .new_text import generate_response
from .get_embedding import get_embedding, get_embeddings_in_batch

def sample(client, desired_bit, history, hash_fn, system_prompt, max_length):
    sampled_bit = None
    count = 0
    while sampled_bit != desired_bit:
        # sample message
        message = generate_response(client, history, system_prompt, max_length)
        emb = get_embedding(client, message)
        # Reshape the embedding to be a 2D array with one row
        emb = np.array(emb).reshape(1, -1)
        print('message:', message)
        sampled_bit = hash_fn(emb)
        print(sampled_bit)
        count = count + 1
    return message

def sample_concurrent(
    client, 
    desired_bits,  # List of bits (the chunk)
    history, 
    hash_fn, 
    k=5,
    system_prompt="You are having a casual conversation.",
    max_length=200
):
    sampled_bits = None
    
    while not np.array_equal(sampled_bits, desired_bits):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Step 1: Parallelize `generate_response`
            response_futures = [
                executor.submit(
                    generate_response, 
                    client, 
                    history,
                    system_prompt,
                    max_length
                ) for _ in range(k)
            ]
            responses = [future.result() for future in concurrent.futures.as_completed(response_futures)]

            # Step 2: Get embeddings in batch
            embeddings = get_embeddings_in_batch(client, responses)

            # Process embeddings
            for message, emb in zip(responses, embeddings):
                emb = np.array(emb).reshape(1, -1)
                sampled_bits = hash_fn(emb)
                
                print('message:', message)
                print('sampled_bits:', sampled_bits)
                print('desired_bits:', desired_bits)
                
                #! Ensure matching shapes for all combinations of inputs and settings
                if np.array_equal(sampled_bits, desired_bits):
                    return message

def encode(client, chunks, history, hash_fn, k=5, system_prompt="You are having a casual conversation.", max_length=200):
    cover_text = []
    for chunk in chunks:
        response = sample_concurrent(
            client, 
            chunk,
            history, 
            hash_fn,
            k=k,
            system_prompt=system_prompt,
            max_length=max_length
        )
        history.append(response)
        cover_text.append(response)
    return cover_text

def encode_resume(
    client, 
    m, 
    history_start, 
    hash_fn, 
    cover_text_partial, 
    starting_idx,
    k=5,
    system_prompt="You are having a casual conversation.",
    max_length=200
):
    history = history_start + cover_text_partial
    cover_text = cover_text_partial
    for idx, bit in enumerate(m[starting_idx:]):
        logging.info((history, cover_text, m, starting_idx + idx, bit))
        response = sample_concurrent(
            client, 
            bit, 
            history, 
            hash_fn,
            k,
            system_prompt=system_prompt,
            max_length=max_length
        )
        history.append(response)
        cover_text.append(response)
    return cover_text
