from abc import ABC, abstractmethod
import torch

class PerturbFunction(ABC):
    """Base class for probability distribution perturbation functions."""
    
    @abstractmethod
    def __call__(self, p, r, delta):
        """
        Perturb probability distribution p based on PRF output r.
        
        Args:
            p: Original probability distribution (torch.Tensor)
            r: PRF output bits (list of 0/1)
            delta: Perturbation strength parameter (float)
            
        Returns:
            Perturbed probability distribution (torch.Tensor)
        """
        pass
