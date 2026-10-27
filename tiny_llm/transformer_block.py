import numpy as np
from tiny_llm.attention import MultiHeadAttention
from tiny_llm.activations import gelu

class LayerNorm:
    \"\"\"
    Layer Normalization over the last feature dimension:
    LN(x) = ((x - mean) / sqrt(var + eps)) * gamma + beta
    \"\"\"
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement layer normalization
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class FeedForward:
    \"\"\"
    Position-wise Feed-Forward Network:
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    \"\"\"
    def __init__(self, d_model: int, d_ff: int):
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement FFN forward pass with GELU
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

class TransformerBlock:
    \"\"\"
    Pre-LayerNorm Transformer Block:
    x = x + MHA(LN(x))
    x = x + FFN(LN(x))
    \"\"\"
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray, is_causal: bool = True) -> np.ndarray:
        # TODO: Implement pre-LN residual transformer block
        pass

    def __call__(self, x: np.ndarray, is_causal: bool = True) -> np.ndarray:
        return self.forward(x, is_causal=is_causal)
