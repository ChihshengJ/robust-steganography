from abc import ABC, abstractmethod

class CovertextCalculator(ABC):
    """Base class for calculating required covertext length."""
    
    @abstractmethod
    def get_covertext_length(self, n: int, epsilon: float, delta: float, p0: float = 0.5) -> int:
        """
        Compute minimum required covertext length for reliable message recovery.
        
        Parameters
        ----------
        n : int
            Number of bits in the hidden message.
        epsilon : float
            Maximum overall error probability.
        delta : float
            Watermark perturbation strength.
        p0 : float, optional
            Null probability that a token is labeled '1', by default 0.5.
            
        Returns
        -------
        int
            Required covertext length (number of tokens).
        """
        pass 