import numpy as np
import json
import os
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

# read in json data from a file
def read_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# 'all' is a list of strings used for many measurement tests initially
# create and return 'all'
def create_all_text():
    text = read_data('./text.json')
    short = text['ai_text']['short']['text']
    medium = text['ai_text']['medium']['text']
    long = text['ai_text']['long']['text']
    all = short + medium + long
    return all

# plot a heatmap of a similarity matrix
def plot_heatmap(matrix, measurement, normalize, comparison, engine):
    plt.figure()
    ax = sns.heatmap(matrix, linewidth=0.5, cbar=True)
    ax.set_title(f'{comparison} norm={normalize} {measurement} Similarity')
    save_plot(ax, f'{comparison}_norm_{normalize}_{measurement}', engine)

def save_plot(ax, measurement, model_name):
    model_name = model_name.replace('/', '_')
    filename = f'./embedding_comparison/{model_name}/{measurement}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig = ax.get_figure()
    fig.savefig(filename)

# save a (text, embedding) pair to a file
def save_embeddings(embeddings, text, label, normalize, model_name):
    model_name = model_name.replace('/', '_')
    filename = f'./embeddings/{model_name}/{label}_norm_{normalize}.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert the NumPy array to a list
    embedding_list = [e.tolist() for e in embeddings]

    # Create a dictionary to store the data
    data = {
        "texts": text,
        "embeddings": embedding_list,
        "normalized": normalize,
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def save_cutoffs(cutoff, error_rate, measurement, comparison, normalize, model_name):
    model_name = model_name.replace('/', '_')
    filename = f'./embedding_comparison/{model_name}/{comparison}_norm_{normalize}_{measurement}_cutoffs.txt'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write(f'{cutoff}\n')
        f.write(f'{error_rate}\n')
        f.write(f'{normalize}\n')

def find_cutoff(matrix, error_rate):
    matrix = np.array(matrix)
    n = matrix.shape[0]
    
    # Get diagonal and off-diagonal elements
    diagonal = list(np.diagonal(matrix))
    off_diagonal = list(matrix[~np.eye(n, dtype=bool)])
    
    cutoff = find_cutoff_helper(diagonal, off_diagonal, error_rate)
    
    return cutoff

def find_cutoff_helper(little, big, error_rate):
    
    if little == [] or big == []:
        return None
    
    # Step 1: Sort both lists
    little_sorted = sorted(little)
    big_sorted = sorted(big)

    # Step 2: Check for no overlap
    if little_sorted[-1] < big_sorted[0]:
        return (little_sorted[-1] + big_sorted[0]) / 2

    # Create a list of potential cutoff points from data points and midpoints between them
    potential_cutoffs = sorted(set(little_sorted + big_sorted))
    cutoffs_to_test = []
    for i in range(len(potential_cutoffs) - 1):
        cutoffs_to_test.append((potential_cutoffs[i] + potential_cutoffs[i + 1]) / 2)

    # Step 3: Find the best cutoff with minimal error rate
    min_error = float('inf')
    best_cutoff = None
    
    for cutoff in cutoffs_to_test:
        # Calculate the error rate for the current cutoff
        error_little = sum(1 for x in little_sorted if x >= cutoff) / len(little_sorted)
        error_big = sum(1 for x in big_sorted if x <= cutoff) / len(big_sorted)
        total_error = error_little + error_big
        
        # Check if this is the best cutoff found so far within acceptable error_rate
        if total_error < min_error and total_error <= error_rate:
            min_error = total_error
            best_cutoff = cutoff

    # Step 4: Return the best cutoff found or None if no acceptable cutoff exists
    return best_cutoff if best_cutoff and min_error <= error_rate else None

# measure similarity of embeddings over 2 sets of texts
def plot_embedding_similarities(embeddings, normalize, engine, comparison):
    # Compute similarities
    cosine_matrix = cdist(embeddings[0], embeddings[1], metric='cosine')
    euclidean_matrix = cdist(embeddings[0], embeddings[1], metric='euclidean')

    # Plot similarities
    plot_heatmap(cosine_matrix, 'cosine', normalize, comparison, engine)
    plot_heatmap(euclidean_matrix, 'euclidean', normalize, comparison, engine)

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
def get_embedding(text, model, client):
    # Using the embeddings.create method to fetch the embedding
    response = client.embeddings.create(
        input=[text],  # Ensure input is a list of text
        model=model    # Specify the model you are using
    )
    # Extracting the embedding from the response object
    embedding = response.data[0].embedding
    return embedding

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
