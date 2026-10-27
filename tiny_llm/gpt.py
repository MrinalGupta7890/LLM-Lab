import numpy as np
from typing import List, Union
from tiny_llm.transformer_block import TransformerBlock, LayerNorm
from tiny_llm.embeddings import EmbeddingTable
from tiny_llm.linear import Linear

class GPT:
    \"\"\"
    A decoder-only autoregressive generative Transformer (GPT).
    \"\"\"
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 128,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 256
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Token and positional embeddings
        self.token_embeddings = EmbeddingTable(vocab_size, d_model)
        self.pos_embeddings = EmbeddingTable(max_seq_len, d_model)
        
        # Stack of Transformer blocks
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        
        # Final LayerNorm and unembedding projection
        self.ln_final = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        \"\"\"
        Args:
            input_ids: Array of token IDs with shape (batch_size, seq_len)
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        \"\"\"
        # TODO: Implement GPT forward pass:
        # 1. token_emb + pos_emb
        # 2. pass through each TransformerBlock
        # 3. ln_final
        # 4. lm_head -> logits
        pass

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        return self.forward(input_ids)
