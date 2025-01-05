import logging
import concurrent.futures
import numpy as np
from .new_text import generate_response
from .get_embedding import get_embedding, get_embeddings_in_batch

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
