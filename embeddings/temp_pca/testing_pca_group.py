import logging
logging.basicConfig(filename='./testing_pca_group.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import sys
from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np
from embeddings.temp_pca.pca_hash_model import load_pca, pca_hash, load_pickled_dataset
from embeddings.temp_pca.testing_pca_hash import load_test_data, hash_to_string
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def split_into_groups(arr, group_size):
#     # Split the array into chunks of the specified group size
#     groups = [arr[i:i + group_size] for i in range(0, len(arr), group_size)]
#     return np.array(groups)

def split_into_groups(arr, groups):
    """
    Split the array into the specified groups
    
    Args:
    - arr (np.ndarray): The input array to split.
    - groups ([(int, int)]): A list of tuples containing the start and end indices of each group.
    """
    # Split the array into chunks of the specified group size
    group_list = [np.array(arr[start:end]) for start, end in groups]
    return group_list
    
    

# def majority(arr):
#     # Count the occurrences of 1s and 0s
#     ones_count = np.sum(arr, axis=-1)  # Sum along the last axis (group level)
#     zeros_count = arr.shape[-1] - ones_count
    
#     # Return 1 if more 1s, else return 0 (vectorized)
#     return np.where(ones_count > zeros_count, 1, 0)

def majority(arr):
    """
    Compute the majority element (0 or 1) for each group, even if the groups have different sizes.
    
    Args:
    - arr (list of np.ndarray): A list of 1D arrays where each array represents a group.

    Returns:
    - np.ndarray: A binary array representing the majority element for each group.
    """
    majority_result = []
    for group in arr:
        ones_count = np.sum(group)  # Count the number of 1s in the group
        zeros_count = len(group) - ones_count  # Count the number of 0s in the group
        
        # Append the majority (1 if more ones, else 0)
        majority_result.append(1 if ones_count > zeros_count else 0)
    
    return np.array(majority_result)

def pca_hash_group(pca, vector, index_map, groups, start, end):
    """
    Compute the majority hash value for a given vector using PCA
    
    Args:
    - pca (PCA): The trained PCA object.
    - vector (np.ndarray): The input vector to hash.
    - index_map (np.ndarray): The permutation of indices to apply to the vector.
    - start (int): The starting index for the hash (inclusive).
    - end (int): The ending index for the hash (inclusive).
    #! Todo: Change end to be exclusive (original pca_hash function)
    #! Todo: majority could amplify any bias in pca_hash -> further testing needed
    """
    
    vector = np.array(vector).reshape(1, -1)
    
    original_hash = pca_hash(pca, vector, start, end)
    
    # Flatten the array and convert to integers
    original_hash = original_hash.flatten().astype(int)
    
    # permute the indices of the hash according to index_map
    permuted_hash = original_hash[index_map]

    # Use numpy.array_split to split the array
    chunked_hash = split_into_groups(permuted_hash, groups)
    
    majority_hash = majority(chunked_hash)
    
    return majority_hash
    
def test_examples_embeddings(pca, index_map, paraphrase_embedding_pairs, groups, start, end, verbose=False):
    buckets = defaultdict(int)
    matches = 0
    
    for emb1, emb2 in tqdm(paraphrase_embedding_pairs):
        emb1 = np.array(emb1).reshape(1, -1)
        emb2 = np.array(emb2).reshape(1, -1)
        hash_code1 = hash_to_string(pca_hash_group(pca, emb1, index_map, groups, start, end))
        hash_code2 = hash_to_string(pca_hash_group(pca, emb2, index_map, groups, start, end))
        # print(hash_code1)
        buckets[hash_code1] += 1
        if hash_code1 == hash_code2:
            if verbose:
                logging.info("The hash codes are identical.")
            matches += 1
        elif verbose:
                logging.info("The hash codes are different.")
        if verbose:
            logging.info(f"Text 1: {emb1}")
            logging.info(f"Text 2: {emb2}")
            logging.info(f"Hash code for Text 1: {hash_code1}")
            logging.info(f"Hash code for Text 2: {hash_code2}")
            
    logging.info(f"Number of matches: {matches / len(paraphrase_embedding_pairs)}")
    for hash_code, count in buckets.items():
        logging.info(f"Hash code: {hash_code}, Count: {count}")
        
def test_consistency_embeddings(pca, index_map, groups, start, end):
    embeddings1, embeddings2 = load_test_data()
    pairs = list(zip(embeddings1, embeddings2))
    test_examples_embeddings(pca, index_map, pairs, groups, start, end, verbose=False)
    
if __name__ == "__main__":
    # (3, 0, 9) -> 0.84488
    # (5, 0, 15) -> 0.78274
    pca = load_pca()
    groups = [(0, 1), (1, 4), (4, 7)]
    start = 0
    end = 7
    index_map = np.arange(end - start)
    # index_map = np.random.permutation(end - start)
    test_consistency_embeddings(pca, index_map, groups, start, end)
