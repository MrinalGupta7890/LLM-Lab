import sys
import numpy as np
from micro_gpt.attention import scaled_dot_product_attention

def test_scaled_dot_product():
    seq_len, d_k, d_v = 4, 8, 8
    q = np.random.randn(1, seq_len, d_k)
    k = np.random.randn(1, seq_len, d_k)
    v = np.random.randn(1, seq_len, d_v)
    
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out is not None, "scaled_dot_product_attention returned None"
    assert out.shape == (1, seq_len, d_v), f"Expected output shape (1, 4, 8), got {out.shape}"
    assert weights.shape == (1, seq_len, seq_len), f"Expected weights shape (1, 4, 4), got {weights.shape}"
    assert np.allclose(np.sum(weights, axis=-1), 1.0), "Attention weights must sum to 1.0"
    print("test_scaled_dot_product passed!")

if __name__ == '__main__':
    try:
        test_scaled_dot_product()
        print("ALL TESTS PASSED: Sprint 3 Ticket 1")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
