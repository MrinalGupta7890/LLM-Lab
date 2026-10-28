import sys
import numpy as np
from micro_gpt.activations import softmax

def test_softmax_stability():
    logits = np.array([1000.0, 1001.0, 1002.0])
    probs = softmax(logits)
    assert probs is not None, "softmax() returned None"
    assert not np.isnan(probs).any(), "softmax produced NaN"
    assert np.isclose(np.sum(probs), 1.0), f"Probabilities do not sum to 1: {np.sum(probs)}"
    
    batch_logits = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    batch_probs = softmax(batch_logits, axis=-1)
    assert np.allclose(np.sum(batch_probs, axis=-1), [1.0, 1.0]), "Batch probabilities do not sum to 1"
    print("test_softmax_stability passed!")

if __name__ == '__main__':
    try:
        test_softmax_stability()
        print("ALL TESTS PASSED: Sprint 2 Ticket 1")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
