import sys
import numpy as np
from tiny_llm.attention import MultiHeadAttention

def test_multi_head_attention():
    batch, seq_len, d_model, num_heads = 2, 4, 16, 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = np.random.randn(batch, seq_len, d_model)
    out = mha(x, is_causal=True)
    assert out is not None, "MultiHeadAttention forward returned None"
    assert out.shape == (batch, seq_len, d_model), f"Expected shape (2, 4, 16), got {out.shape}"
    print("test_multi_head_attention passed!")

if __name__ == '__main__':
    try:
        test_multi_head_attention()
        print("ALL TESTS PASSED: Sprint 3 Ticket 3")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
