import pickle
from sklearn.decomposition import PCA
import numpy as np

def train_pca_model(vectors, n_components=None):
    """
    Train a PCA model on the given vectors and return the PCA object.
    
    Args:
        vectors (np.ndarray): Input vectors of shape (num_samples, num_features).
        n_components (int, optional): Number of components to keep. If None, keep all components.
    
    Returns:
        PCA: Trained PCA model.
    """
    pca = PCA(n_components=n_components)
    pca.fit(vectors)
    return pca

def save_pca_model(pca_model, filename):
    """
    Save the PCA model to a file using pickle.
    
    Args:
        pca_model (PCA): The trained PCA model.
        filename (str): The filename to save the PCA model.
    """
    with open(filename, 'wb') as f:
        pickle.dump(pca_model, f)
    print(f"PCA model saved to {filename}")

def load_pca_model(filename):
    """
    Load a PCA model from a file.
    
    Args:
        filename (str): The filename of the saved PCA model.
    
    Returns:
        PCA: The loaded PCA model.
    """
    with open(filename, 'rb') as f:
        pca_model = pickle.load(f)
    print(f"PCA model loaded from {filename}")
    return pca_model
