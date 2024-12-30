import numpy as np
from math import ceil, sqrt
from scipy.stats import norm
from .base import CovertextCalculator

class SmoothCovertextCalculator(CovertextCalculator):
    """Calculator for smooth perturbation watermarking (only the 1-bits are used)."""
    
    def get_covertext_length(self, n: int, epsilon: float, delta: float, p0: float = 0.5) -> int:
        """
        Computes the minimum required covertext length L so that
        an n-bit message can be recovered with overall success probability ≥ 1 - epsilon,
        in the scenario where only bits that are actually '1' are used (chosen uniformly
        among themselves). If all n bits were 1, that worst-case uses k_max = n.

        Parameters
        ----------
        n : int
            Number of bits in the hidden message.
        epsilon : float
            Maximum overall error probability (probability that any bit is incorrectly recovered).
            For example, epsilon=0.05 for a 95% chance of recovering all n bits correctly.
        delta : float
            Watermark perturbation strength.
        p0 : float, optional
            Null (unwatermarked) probability that a token is labeled '1', by default 0.5.

        Returns
        -------
        int
            The required covertext length L (number of tokens), rounded up.

        Explanation
        -----------
        1. We assume the worst case: all n bits might be 1, so each bit is chosen with probability 1/n.
           Then the fraction of '1'-labeled tokens for a watermarked bit is 0.5 + (p_w - 0.5)/n.
        2. A binomial-proportion Z-test requires:
               2 * sqrt(L) * [(p_w - 0.5)/n]  >  z
           to exceed the threshold z for high detection probability.
        3. Rearranging yields:
               L > ( z^2 * n^2 ) / [ 4 * (p_w - 0.5)^2 ].
        4. We get z from a per-bit error of alpha_bit = epsilon / n, to keep total error ≤ epsilon.
        """
        # Basic validation
        if n <= 0:
            raise ValueError("Message length n must be positive")
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError("Error probability epsilon must be in (0,1)")
        if delta <= 0:
            raise ValueError("Perturbation strength delta must be positive")
        if p0 <= 0 or p0 >= 1:
            raise ValueError("Null probability p0 must be in (0,1)")

        # Convert overall epsilon to per-bit alpha via union bound
        alpha_bit = epsilon / n

        # z-score from the Normal distribution (inverse CDF for the per-bit error).
        z = norm.ppf(1 - alpha_bit)

        # Watermarked probability (assuming p0=0.5).
        # p_w = 2*(1+delta)/(4+delta)
        p_w = (2.0 * (1.0 + delta)) / (4.0 + delta)
        shift = p_w - p0  # (p_w - 0.5)

        # Formula: L >= [z^2 * n^2] / [4 * (p_w - 0.5)^2]
        # Avoid dividing by zero if p_w == 0.5 (which won't happen if delta>0).
        if abs(shift) < 1e-12:
            raise ValueError("delta too small or p_w too close to 0.5 for meaningful detection.")

        L_float = (z**2 * n**2) / (4.0 * (shift**2))
        L = ceil(L_float)  # round up

        return L
