import torch
from .base import PerturbFunction

class HarshPerturb(PerturbFunction):
    """
    Harsh perturbation that zeros out red list entries and increases green list probabilities.
    This is more aggressive than DeltaPerturb and may produce less natural text but stronger watermarks.
    """
    
    def __call__(self, p: torch.Tensor, r: list, delta: float) -> torch.Tensor:
        """
        Perturb probability distribution by zeroing red list and boosting green list.
        
        Args:
            p: Original probability distribution
            r: PRF output bits (list of 0/1)
            delta: Perturbation strength parameter
            
        Returns:
            Perturbed probability distribution
        """
        # Create mask for valid probabilities (within [2δ, 1-2δ])
        valid_mask = (p >= 2*delta) & (p <= 1 - 2*delta)
        
        # Zero out all probabilities first
        perturbed = torch.zeros_like(p)
        
        # Set probabilities for indices where r[i] = 1
        for idx in range(len(p)):
            if r[idx] == 1:
                if valid_mask[idx]:
                    perturbed[idx] = p[idx] + delta
                else:
                    perturbed[idx] = p[idx]
                
        # Normalize to make it a valid distribution
        if torch.sum(perturbed) > 0:
            perturbed = perturbed / torch.sum(perturbed)
            
        return perturbed 