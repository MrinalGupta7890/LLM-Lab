import sys
import numpy as np
from micro_gpt.embeddings import vector_dot, cosine_similarity

def test_vector_ops():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    
    dot = vector_dot(v1, v2)
    assert dot is not None, "vector_dot returned None. Implement it in micro_gpt/embeddings.py"
    assert np.isclose(dot, 32.0), f"Expected 32.0, got {dot}"
    
    sim_same = cosine_similarity(v1, v1)
    assert np.isclose(sim_same, 1.0), f"Expected 1.0 for identical vectors, got {sim_same}"
    
    ortho1 = np.array([1.0, 0.0])
    ortho2 = np.array([0.0, 1.0])
    sim_ortho = cosine_similarity(ortho1, ortho2)
    assert np.isclose(sim_ortho, 0.0), f"Expected 0.0 for orthogonal vectors, got {sim_ortho}"
    print("test_vector_ops passed!")

if __name__ == '__main__':
    try:
        test_vector_ops()
        print("ALL TESTS PASSED: Sprint 1 Ticket 1")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
