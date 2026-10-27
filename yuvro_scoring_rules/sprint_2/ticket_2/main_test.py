import sys
import numpy as np
from tiny_llm.activations import softmax

def test_temperature_scaling():
    logits = np.array([2.0, 1.0, 0.0])
    cold_probs = softmax(logits, temperature=0.01)
    assert np.argmax(cold_probs) == 0
    assert cold_probs[0] > 0.99, f"Cold temperature should be near 1.0 at max, got {cold_probs[0]}"
    
    hot_probs = softmax(logits, temperature=100.0)
    assert np.allclose(hot_probs, [0.333, 0.333, 0.333], atol=0.02), f"Hot temperature should be near uniform"
    print("test_temperature_scaling passed!")

if __name__ == '__main__':
    try:
        test_temperature_scaling()
        print("ALL TESTS PASSED: Sprint 2 Ticket 2")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
