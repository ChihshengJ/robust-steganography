import torch
import numpy as np
from .base import PerturbFunction

class DeltaPerturb(PerturbFunction):
    """
    Implementation of delta-based probability perturbation.
    Adjusts token probabilities up/down based on PRF output.
    """
    
    def __call__(self, p, r, delta):
        """
        p: A 1D torch.Tensor or numpy array of probabilities for each token.
        r: A list (or array) of 0/1 bits from PRF_t for each token.
        delta: Strength parameter controlling how much to boost/suppress probabilities.

        Returns a modified probability distribution p' that is renormalized.
        """
        N = len(p)
        w = sum(r)  # Number of tokens allowed by watermark
        if w == 0 or w == N:
            # Edge case: if r is all 0 or all 1, fall back to a small shift or no shift
            return p  # or slightly perturb if you want

        # Example multipliers. Adjust to taste:
        alpha = 1.0 + delta      # Factor by which to multiply tokens with r=1
        beta  = 1.0 - delta/2.0  # Factor for tokens with r=0 (less severe penalty)

        # Apply each multiplier
        for j in range(N):
            if r[j] == 1:
                p[j] = p[j] * alpha
            else:
                p[j] = p[j] * beta
            # Ensure no negatives (just in case p is super small):
            if p[j] < 0:
                p[j] = 0

        # Renormalize
        total = p.sum() if hasattr(p, 'sum') else sum(p)
        if total > 0:
            p = p / total
        else:
            # Fallback to uniform if numerical underflow occurred
            p = p.new_ones(N) / float(N) if hasattr(p, 'new_ones') else np.ones(N) / float(N)

        return p
