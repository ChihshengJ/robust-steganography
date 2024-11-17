import logging
logging.basicConfig(filename='./testing_pca.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import sys
from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np
from embeddings.temp_pca.pca_hash_model import load_pca, pca_hash, load_pickled_dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_training_data():
    embeddings1_pos, embeddings2_pos = load_pickled_dataset()
    embeddings1_pos = embeddings1_pos[:-50000]
    embeddings2_pos = embeddings2_pos[:-50000]
    return embeddings1_pos, embeddings2_pos

def load_test_data():
    embeddings1_pos, embeddings2_pos = load_pickled_dataset()
    embeddings1_pos = embeddings1_pos[-50000:]
    embeddings2_pos = embeddings2_pos[-50000:]
    return embeddings1_pos, embeddings2_pos

def load_random_paragraphs_data():
    embeddings1 = np.load('random_paragraphs_embeddings.npy')
    embeddings2 = np.load('random_paraphrases_embeddings.npy')
    return embeddings1, embeddings2

def hash_to_string(hash_array):
    """
    Convert a NumPy array of binary values (0s and 1s) into a string of '1's and '0's.
    
    Args:
    - hash_array (np.ndarray): The input binary array, shape (1, n_bits).
    
    Returns:
    - hash_string (str): A string representation of the hash (e.g., '0101').
    """
    # Flatten the array and convert to integers
    hash_flat = hash_array.flatten().astype(int)
    
    # Join the bits into a string
    hash_string = ''.join(map(str, hash_flat))
    
    return hash_string

def test_examples_embeddings(pca, paraphrase_embedding_pairs, start, end, verbose=False):
    buckets = defaultdict(int)
    matches = 0
    
    for emb1, emb2 in tqdm(paraphrase_embedding_pairs):
        if verbose:
            logging.info(f"Text 1: {emb1}")
            logging.info(f"Text 2: {emb2}")
        emb1 = np.array(emb1).reshape(1, -1)
        emb2 = np.array(emb2).reshape(1, -1)
        hash_code1 = hash_to_string(pca_hash(pca, emb1, start, end))
        hash_code2 = hash_to_string(pca_hash(pca, emb2, start, end))
        # print(hash_code1)
        buckets[hash_code1] += 1
        logging.info(f"Hash code for Text 1: {hash_code1}")
        logging.info(f"Hash code for Text 2: {hash_code2}")
        if hash_code1 == hash_code2:
            logging.info("The hash codes are identical.")
            matches += 1
        else:
            logging.info("The hash codes are different.")
            
    logging.info(f"Number of matches: {matches / len(paraphrase_embedding_pairs)}")
    for hash_code, count in buckets.items():
        logging.info(f"Hash code: {hash_code}, Count: {count}")
        
def test_consistency_embeddings(pca, start, end):
    embeddings1, embeddings2 = load_test_data()
    pairs = list(zip(embeddings1, embeddings2))
    test_examples_embeddings(pca, pairs, start, end, verbose=True)

def plot_pca_test_data(pca, embeddings, start=0, end=2):
    """
    Plot the PCA projections of the test data (embeddings) using components 
    between `start` and `end` indices.

    Args:
    - pca (PCA): Trained PCA object.
    - embeddings (np.ndarray): The input embeddings, shape (num_samples, num_features).
    - start (int): The starting index for PCA components.
    - end (int): The ending index for PCA components.
    """
    # Project embeddings using PCA
    transformed_embeddings = pca.transform(embeddings)
    
    # Select PCA components between `start` and `end`
    selected_components = transformed_embeddings[:, start:end]

    # 1D Plot as a histogram if only 1 component is selected
    if end - start == 1:
        plt.hist(selected_components[:, 0], bins=30, alpha=0.75)
        plt.title(f'PCA Component {start} Distribution')
        plt.xlabel(f'Component {start} Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # 2D Plot if 2 components are selected
    elif end - start == 2:
        plt.scatter(selected_components[:, 0], selected_components[:, 1], alpha=0.5)
        plt.title(f'PCA Components {start} and {end-1}')
        plt.xlabel(f'Component {start}')
        plt.ylabel(f'Component {end-1}')
        plt.grid(True)
        plt.show()

    # 3D Plot if 3 components are selected
    elif end - start == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(selected_components[:, 0], selected_components[:, 1], selected_components[:, 2], alpha=0.5)
        ax.set_title(f'PCA Components {start} to {end-1}')
        ax.set_xlabel(f'Component {start}')
        ax.set_ylabel(f'Component {start+1}')
        ax.set_zlabel(f'Component {end-1}')
        plt.show()

    else:
        print("Can only visualize 1, 2, or 3 components for a scatter or line plot.")


def plot_test_embeddings(dataset='test'):
    """
    Load test data and plot the PCA components of embeddings.
    """
    # Load the PCA model
    pca = load_pca()

    if dataset == 'test':
        # Load the test embeddings
        embeddings1, embeddings2 = load_test_data()
    elif dataset == 'train':
        # Load the training embeddings
        embeddings1, embeddings2 = load_training_data()
    else:
        raise ValueError("Dataset must be either 'train' or 'test'.")

    # Concatenate the embeddings for PCA
    test_embeddings = np.concatenate([embeddings1, embeddings2], axis=0)

    # Plot the PCA for 1 component (1D)
    plot_pca_test_data(pca, test_embeddings, start=0, end=1)
    
    # Plot the PCA for 2 components (2D)
    plot_pca_test_data(pca, test_embeddings, start=0, end=2)
    
    # Optional: Plot for the next set of components (2nd and 3rd, 3D)
    plot_pca_test_data(pca, test_embeddings, start=0, end=3)

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python testing_pca.py [start] [end]")
        sys.exit(1)

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
    except ValueError:
        print("Both start and end should be integers.")
        sys.exit(1)
    
    pca = load_pca()
    test_consistency_embeddings(pca, start, end)
    
    # plot_test_embeddings('test')