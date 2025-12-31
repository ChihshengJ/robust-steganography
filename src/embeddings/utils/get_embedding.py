import numpy as np
from openai import OpenAI
import requests
import os
import time

API_BASE = "https://api.openai.com/v1"
API_KEY = os.getenv("OPENAI_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def get_embedding(text):
    payload = {
        "model": "text-embedding-3-large",
        "input": text,
    }

    r = requests.post(
        f"{API_BASE}/embeddings",
        headers=HEADERS,
        json=payload,
        timeout=60,
    )
    r.raise_for_status()

    data = r.json()["data"]
    embeddings = [d["embedding"] for d in data]

    return np.array(embeddings[0])


def get_embeddings_in_batch(client, texts):
    # Using the embeddings.create method to fetch embeddings for multiple texts in one request
    response = client.embeddings.create(
        input=texts,  # Input is a list of texts
        model="text-embedding-3-large"   # Specify the model you are using
    )
    # Extracting the embeddings from the response object
    embeddings = np.array([res.embedding for res in response.data])
    return embeddings

if __name__ == "__main__":
    pass
