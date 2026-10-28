import numpy as np
from typing import Optional, Tuple
from micro_gpt.activations import softmax

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create an upper-triangular causal mask where positions to be ignored
    are filled with -1e9 (-inf) and valid positions are 0.0.
    
    Args:
        seq_len: Length of the sequence
    Returns:
        Mask matrix of shape (seq_len, seq_len)
    """
    # TODO: Implement causal mask with np.triu
    pass

def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k) + mask) @ V
    
    Args:
        q: Query tensor of shape (..., seq_len_q, d_k)
        k: Key tensor of shape (..., seq_len_k, d_k)
        v: Value tensor of shape (..., seq_len_v, d_v)
        mask: Optional mask tensor of shape (..., seq_len_q, seq_len_k)
    Returns:
        output: Contextualized output of shape (..., seq_len_q, d_v)
        attention_weights: Attention distribution weights of shape (..., seq_len_q, seq_len_k)
    """
    # TODO: Implement scaled dot-product attention
    pass

class MultiHeadAttention:
    """
    Multi-Head Attention mechanism dividing hidden dimension d_model across num_heads.
    """
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_k = self.head_dim

        # Projections
        scale = np.sqrt(1.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, x: np.ndarray, is_causal: bool = True) -> np.ndarray:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            is_causal: If True, apply causal lower-triangular mask
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # TODO: Implement Multi-Head Attention forward pass
        pass

    def __call__(self, x: np.ndarray, is_causal: bool = True) -> np.ndarray:
        return self.forward(x, is_causal=is_causal)
