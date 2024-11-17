import importlib.util
from tqdm import tqdm
import json
import pandas as pd
from openai import OpenAI
from embedding_utils import compute_embeddings

def get_embedding(client, text):
    # Get the embedding for the text
    embedding = compute_embeddings(
        text, True, "text-embedding-3-large", client)
    return embedding[0]

def get_embeddings_in_batch(client, texts):
    # Using the embeddings.create method to fetch embeddings for multiple texts in one request
    response = client.embeddings.create(
        input=texts,  # Input is a list of texts
        model="text-embedding-3-large"   # Specify the model you are using
    )
    # Extracting the embeddings from the response object
    embeddings = [res.embedding for res in response.data]
    return embeddings

if __name__ == "__main__":
    client = OpenAI()  # automatically uses OPENAI_API_KEY env var
    text = "What are you up to today?"
    embedding = get_embedding(client, text)
    print(embedding)
    print(type(embedding))
