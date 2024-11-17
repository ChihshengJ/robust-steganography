import numpy as np
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from tabulate import tabulate
from itertools import product

# Load the pickled dataset
def load_pickled_dataset(filename='dataset.pkl'):
    """
    Load the embeddings dataset from a pickle file.
    
    Args:
    - filename (str): Path to the pickled dataset.
    
    Returns:
    - embeddings1 (np.ndarray): The first set of embeddings.
    - embeddings2 (np.ndarray): The second set of embeddings.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings1'], data['embeddings2']

def save_pca(pca, filename='pca_model.pkl'):
    """
    Save the PCA object to a file using pickle.
    
    Args:
    - pca (PCA): The trained PCA object.
    - filename (str): The filename to save the PCA model.
    """
    with open(filename, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA model saved to {filename}")

def load_pca(filename='pca_model.pkl'):
    """
    Load the PCA object from a file.
    
    Args:
    - filename (str): The filename of the saved PCA model.
    
    Returns:
    - pca (PCA): The loaded PCA object.
    """
    with open(filename, 'rb') as f:
        pca = pickle.load(f)
    print(f"PCA model loaded from {filename}")
    return pca

# Perform PCA on concatenated embeddings
def compute_full_pca(vectors, n_components=None):
    """
    Perform PCA on the given vectors and return the PCA object.
    
    Args:
    - vectors (np.ndarray): The input vectors, shape (num_samples, num_features).
    - n_components (int): Number of components to keep. If None, keep all components.
    
    Returns:
    - pca (PCA): Trained PCA object with the components saved.
    """
    pca = PCA(n_components=n_components)
    pca.fit(vectors)
    return pca

# Hashing function using PCA
def pca_hash(pca, vectors, start=0, end=None):
    """
    Hash the given vectors using the PCA components between `start` and `end` indices.
    
    Args:
    - pca (PCA): The trained PCA object.
    - vectors (np.ndarray): The input vectors to hash, shape (num_samples, num_features).
    - start (int): The starting index for selecting PCA components.
    - end (int): The ending index for selecting PCA components. If None, use all remaining components.
    
    Returns:
    - hashes (np.ndarray): Binarized hash values, shape (num_samples, num_bits).
    """
    # Project the vectors using PCA
    transformed_vectors = pca.transform(vectors)
    
    # Select the range of components
    selected_components = transformed_vectors[:, start:end]
    
    # Binarize by taking the sign of the components
    hashes = np.sign(selected_components)
    
    # Convert -1s to 0s for a binary representation
    hashes = (hashes + 1) // 2
    
    return hashes

def measure_accuracy(pca, hashes1, hashes2, start_index, end_index):
    hashes1 = pca_hash(pca, embeddings1_pos, start=start_index, end=end_index)
    hashes2 = pca_hash(pca, embeddings2_pos, start=start_index, end=end_index)

    # Compare the hashes element-wise and count the matches
    matches = np.sum(np.all(hashes1 == hashes2, axis=1))
    
    # Calculate the percentage of matches
    total_samples = len(hashes1)
    match_percentage = (matches / total_samples) * 100
    
    return match_percentage
    
def measure_uniformity(hashes, num_bits):
    """
    Check how well the output space is distributed by calculating the percentage of 
    each possible binary output (from 0 to 2^b - 1) that the hash values map to.
    
    Args:
    - hashes (np.ndarray): The array of hash values (shape: num_samples, num_bits).
    - num_bits (int): The number of bits in each hash value.
    
    Returns:
    - uniformity (dict): A dictionary where keys are binary values (as strings) and 
                         values are the percentage of samples that mapped to each binary output.
    """
    # Generate all possible binary values of length num_bits
    all_possible_hashes = [''.join(map(str, bits)) for bits in product([0, 1], repeat=num_bits)]
    
    # Convert the actual hashes to strings for comparison
    hash_strings = [''.join(map(str, map(int, h))) for h in hashes]

    # Count how many times each possible hash value appears
    hash_counts = {possible_hash: 0 for possible_hash in all_possible_hashes}
    
    for h in hash_strings:
        if h in hash_counts:
            hash_counts[h] += 1
    
    # Calculate the percentage of occurrences for each possible hash value
    total_hashes = len(hash_strings)
    uniformity = {key: (count / total_hashes) * 100 for key, count in hash_counts.items()}
    
    return uniformity

# Main function to load, concatenate, and apply PCA
if __name__ == "__main__":
    # # Step 1: Load embeddings from the pickled file
    embeddings1_pos, embeddings2_pos = load_pickled_dataset()
    embeddings1_pos = embeddings1_pos[:-50000]
    embeddings2_pos = embeddings2_pos[:-50000]

    # # Step 2: Concatenate the two sets of embeddings
    all_embeddings = np.concatenate([embeddings1_pos, embeddings2_pos], axis=0)

    # # Step 3: Perform PCA on the concatenated embeddings
    pca = compute_full_pca(embeddings1_pos) # all_embeddings
    save_pca(pca)
    exit()

    pca = load_pca()

    # test accuracies for various lenghts and start indices of PCA components
    indices = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 8),
        (2, 9),
        (2, 10),
    ]
    results = []
    for start_index, end_index in tqdm(indices):
        # Measure the accuracy
        match_percentage = measure_accuracy(pca, embeddings1_pos, embeddings2_pos, start_index, end_index)
        # Calculate the uniformity by hashing the embeddings
        hashes = pca_hash(pca, embeddings1_pos, start=start_index, end=end_index)
        uniformity = measure_uniformity(hashes, end_index - start_index)
        results.append([start_index, end_index, f"{match_percentage:.2f}%", uniformity])
    table = tabulate(results, headers=["Start Index", "End Index", "Accuracy (%)"], tablefmt="grid")
    print(table)
    
    # print(measure_uniformity(pca_hash(pca, embeddings1_pos, start=0, end=2), 2))