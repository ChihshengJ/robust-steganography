import numpy as np
import pickle
import argparse
from pathlib import Path

def prepare_dataset(embeddings1_path, embeddings2_path, output_path):
    """
    Prepare dataset for PCA hash training by combining two .npy files
    into a single pickle with the expected format.
    
    Args:
    - embeddings1_path (str): Path to first embeddings .npy file
    - embeddings2_path (str): Path to second embeddings .npy file
    - output_path (str): Path to save the combined pickle file
    """
    # Load both embedding files
    print(f"Loading embeddings from {embeddings1_path}")
    embeddings1 = np.load(embeddings1_path)
    print(f"Shape of embeddings1: {embeddings1.shape}")
    
    print(f"\nLoading embeddings from {embeddings2_path}")
    embeddings2 = np.load(embeddings2_path)
    print(f"Shape of embeddings2: {embeddings2.shape}")
    
    # Verify shapes match
    if embeddings1.shape != embeddings2.shape:
        raise ValueError(f"Embedding shapes don't match: {embeddings1.shape} vs {embeddings2.shape}")
    
    # Create dataset dictionary
    dataset = {
        'embeddings1': embeddings1,
        'embeddings2': embeddings2
    }
    
    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nSaved combined dataset to {output_path}")
    print(f"Total pairs: {len(embeddings1)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for PCA hash training')
    parser.add_argument('--embeddings1', type=str, required=True,
                      help='Path to first embeddings .npy file')
    parser.add_argument('--embeddings2', type=str, required=True,
                      help='Path to second embeddings .npy file')
    parser.add_argument('--output', type=str, default='dataset.pkl',
                      help='Path to save combined pickle file')
    args = parser.parse_args()
    
    prepare_dataset(args.embeddings1, args.embeddings2, args.output) 