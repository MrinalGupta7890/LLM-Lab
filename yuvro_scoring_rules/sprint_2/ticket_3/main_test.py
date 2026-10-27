import sys
import numpy as np
from tiny_llm.linear import Linear, cross_entropy_loss

def test_linear_and_loss():
    layer = Linear(in_features=8, out_features=4)
    x = np.random.randn(2, 3, 8)
    out = layer(x)
    assert out is not None, "Linear forward returned None"
    assert out.shape == (2, 3, 4), f"Expected shape (2, 3, 4), got {out.shape}"
    
    logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    targets = np.array([0, 1])
    loss = cross_entropy_loss(logits, targets)
    assert loss < 0.01, f"Loss for confident correct predictions should be near 0, got {loss}"
    print("test_linear_and_loss passed!")

if __name__ == '__main__':
    try:
        test_linear_and_loss()
        print("ALL TESTS PASSED: Sprint 2 Ticket 3")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
