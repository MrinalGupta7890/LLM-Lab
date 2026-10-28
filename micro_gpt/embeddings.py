import numpy as np
from typing import List, Union

def vector_dot(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the dot product between two 1D vectors.
    
    Args:
        v1: 1D numpy array of shape (d,)
        v2: 1D numpy array of shape (d,)
    Returns:
        Scalar dot product result.
    """
    # TODO: Implement vector dot product using numpy
    pass

def cosine_similarity(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-9) -> float:
    """
    Compute the cosine similarity between two 1D vectors:
    cos_sim(v1, v2) = (v1 . v2) / (||v1|| * ||v2||)
    
    Args:
        v1: 1D numpy array
        v2: 1D numpy array
        eps: Small epsilon to prevent division by zero
    Returns:
        Cosine similarity in range [-1.0, 1.0]
    """
    # TODO: Implement cosine similarity
    pass

class EmbeddingTable:
    """
    Learnable lookup table mapping integer token IDs to continuous feature vectors.
    """
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Initialize random weights with standard normal distribution scaled by 0.02
        self.weights = np.random.randn(vocab_size, d_model) * 0.02

    def lookup(self, token_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Retrieve embedding vectors for given token IDs.
        
        Args:
            token_ids: Array of shape (seq_len,) or (batch_size, seq_len)
        Returns:
            Embeddings array of shape (..., d_model)
        """
        # TODO: Implement embedding lookup using numpy fancy indexing
        pass
