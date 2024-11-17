import numpy as np
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from openai import APIError, Timeout, APIConnectionError, AuthenticationError, RateLimitError

def compute_embeddings_local_pair(texts1, texts2, normalize, engine):
    # Load model
    model = SentenceTransformer(engine)

    # compute embeddings
    embeddings_1 = model.encode(texts1)
    embeddings_2 = model.encode(texts2)
    
    # Normalize embeddings if normalize is True
    if normalize:
        embeddings_1 = [normalize_embedding(e) for e in embeddings_1]
        embeddings_2 = [normalize_embedding(e) for e in embeddings_2]
    
    return [embeddings_1, embeddings_2]

def compute_embeddings_local(texts, normalize, engine):
    # Load model
    model = SentenceTransformer(engine)

    # compute embeddings
    embeddings = model.encode(texts)
    
    # Normalize embeddings if normalize is True
    if normalize:
        embeddings = [normalize_embedding(e) for e in embeddings]
    
    return embeddings

def compute_embeddings(texts, normalize, engine, client):
    embeddings = compute_embeddings_concurrently(texts, engine, client)
    
    # Normalize embeddings if normalize is True
    if normalize:
        embeddings = [normalize_embedding(e) for e in embeddings]
    
    return embeddings

# Get the embedding for a single text from the OpenAI API
# def get_embedding(text, model, client):
#     # Using the embeddings.create method to fetch the embedding
#     response = client.embeddings.create(
#         input=[text],  # Ensure input is a list of text
#         model=model    # Specify the model you are using
#     )
#     # Extracting the embedding from the response object
#     embedding = response.data[0].embedding
#     return embedding

def get_embedding(text, model, client, max_retries=5, backoff_factor=1.5):
    # Initial attempt counter
    attempts = 0
    while attempts < max_retries:
        try:
            # Attempt to fetch the embedding
            response = client.embeddings.create(
                input=[text],  # Ensure input is a list of text
                model=model    # Specify the model you are using
            )
            # Extract and return the embedding from the response object
            embedding = response.data[0].embedding
            return embedding
        
        except (APIError, Timeout, APIConnectionError, AuthenticationError, RateLimitError) as e:
            # Log the error (you could replace this with logging for production)
            print(f"Attempt {attempts+1} failed with error: {e}")
            
            # Increment the attempt counter
            attempts += 1
            
            # If we've exhausted the retries, raise the error
            if attempts == max_retries:
                raise e

            # Exponential backoff: wait longer after each failure
            wait_time = backoff_factor ** attempts
            print(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

# Compute embeddings for texts
# Note that there is no need for a non-concurrent one
def compute_embeddings_concurrently(texts, engine, client):
    # Use ThreadPoolExecutor to parallelize the get_embedding calls to compute the embeddings
    with ThreadPoolExecutor() as executor:
        # Submit all tasks for texts simultaneously
        future_embeddings = executor.map(lambda text: get_embedding(text, engine, client), texts)
        
        # Convert the results to numpy arrays as they become available
        embeddings = np.array(list(future_embeddings))
    
    return embeddings

# Compute embeddings for 2 sets of texts when they are different
def compute_embeddings_pair_concurrently(texts1, texts2, engine, client):
    # Use ThreadPoolExecutor to parallelize the get_embedding calls to compute the embeddings
    with ThreadPoolExecutor() as executor:
        # Submit all tasks for texts1 and texts2 simultaneously
        future_embeddings_1 = executor.map(lambda text: get_embedding(text, engine, client), texts1)
        future_embeddings_2 = executor.map(lambda text: get_embedding(text, engine, client), texts2)
        
        # Convert the results to numpy arrays as they become available
        embeddings_1 = np.array(list(future_embeddings_1))
        embeddings_2 = np.array(list(future_embeddings_2))
    
    return embeddings_1, embeddings_2

# Compute embeddings for 2 sets of texts when they are the same
def compute_embeddings_pair_concurrently_same(texts1, engine, client):
    # Use ThreadPoolExecutor to parallelize the get_embedding calls to compute the embeddings
    with ThreadPoolExecutor() as executor:
        # Submit all tasks for texts1 simultaneously
        future_embeddings_1 = executor.map(lambda text: get_embedding(text, engine, client), texts1)
        
        # Convert the results to numpy arrays as they become available
        embeddings_1 = np.array(list(future_embeddings_1))
        embeddings_2 = embeddings_1
    
    return embeddings_1, embeddings_2

# Compute embeddings for 2 sets of texts
def compute_embeddings_pair(texts1, texts2, normalize, engine, client):
    if texts1 == texts2:
        embeddings_1, embeddings_2 = compute_embeddings_pair_concurrently_same(texts1, engine, client)
    else:
        embeddings_1, embeddings_2 = compute_embeddings_pair_concurrently(texts1, texts2, engine, client)
    
    # Normalize embeddings if normalize is True
    if normalize:
        embeddings_1 = [normalize_embedding(e) for e in embeddings_1]
        embeddings_2 = [normalize_embedding(e) for e in embeddings_2]
    
    return [embeddings_1, embeddings_2]

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)
