import sys
import numpy as np
from micro_gpt.attention import create_causal_mask, scaled_dot_product_attention

def test_causal_mask():
    seq_len = 4
    mask = create_causal_mask(seq_len)
    assert mask is not None, "create_causal_mask returned None"
    assert mask.shape == (seq_len, seq_len), f"Expected shape (4, 4), got {mask.shape}"
    assert mask[0, 1] <= -1e8, "Upper triangle must be masked with negative infinity"
    assert mask[0, 0] == 0.0, "Diagonal must not be masked"
    assert mask[2, 1] == 0.0, "Lower triangle must not be masked"
    print("test_causal_mask passed!")

if __name__ == '__main__':
    try:
        test_causal_mask()
        print("ALL TESTS PASSED: Sprint 3 Ticket 2")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
