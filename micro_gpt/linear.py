import numpy as np
from typing import Optional

class Linear:
    """
    A fully connected linear projection layer: Y = X @ W + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input tensor of shape (..., in_features)
        Returns:
            Output tensor of shape (..., out_features)
        """
        # TODO: Implement linear projection
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute average cross-entropy loss between predicted logits and target class indices.
    
    Args:
        logits: Array of shape (N, vocab_size) or (batch, seq_len, vocab_size)
        targets: Array of shape (N,) or (batch, seq_len) with target class indices
    Returns:
        Scalar loss value
    """
    # TODO: Implement cross-entropy loss
    pass
