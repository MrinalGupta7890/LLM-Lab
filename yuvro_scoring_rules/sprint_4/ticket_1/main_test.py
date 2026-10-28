import sys
import numpy as np
from micro_gpt.transformer_block import LayerNorm

def test_layernorm():
    ln = LayerNorm(d_model=8)
    x = np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]])
    out = ln(x)
    assert out is not None, "LayerNorm returned None"
    assert np.isclose(np.mean(out), 0.0, atol=1e-5), f"LayerNorm mean should be 0, got {np.mean(out)}"
    assert np.isclose(np.var(out), 1.0, atol=1e-2), f"LayerNorm var should be 1, got {np.var(out)}"
    print("test_layernorm passed!")

if __name__ == '__main__':
    try:
        test_layernorm()
        print("ALL TESTS PASSED: Sprint 4 Ticket 1")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
