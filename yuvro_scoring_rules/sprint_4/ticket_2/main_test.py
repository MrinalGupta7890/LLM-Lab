import sys
import numpy as np
from tiny_llm.transformer_block import FeedForward
from tiny_llm.activations import gelu

def test_ffn_and_gelu():
    x = np.array([-2.0, 0.0, 2.0])
    g = gelu(x)
    assert g is not None, "gelu() returned None"
    assert np.isclose(g[1], 0.0), f"GELU(0) should be 0, got {g[1]}"
    
    ffn = FeedForward(d_model=16, d_ff=64)
    inp = np.random.randn(2, 4, 16)
    out = ffn(inp)
    assert out.shape == (2, 4, 16), f"FFN shape expected (2, 4, 16), got {out.shape}"
    print("test_ffn_and_gelu passed!")

if __name__ == '__main__':
    try:
        test_ffn_and_gelu()
        print("ALL TESTS PASSED: Sprint 4 Ticket 2")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
