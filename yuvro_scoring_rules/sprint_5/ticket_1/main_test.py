import sys
import numpy as np
from tiny_llm.sampler import top_k_filtering, top_p_filtering

def test_filtering():
    logits = np.array([1.0, 5.0, 2.0, 10.0, 3.0])
    k_filtered = top_k_filtering(logits.copy(), top_k=2)
    assert k_filtered[3] == 10.0 and k_filtered[1] == 5.0
    assert k_filtered[0] <= -1e8, "Non-top-k logits should be masked out"
    print("test_filtering passed!")

if __name__ == '__main__':
    try:
        test_filtering()
        print("ALL TESTS PASSED: Sprint 5 Ticket 1")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
