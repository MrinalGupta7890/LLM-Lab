import sys
import numpy as np
from tiny_llm.embeddings import EmbeddingTable

def test_embedding_table():
    emb = EmbeddingTable(vocab_size=10, d_model=16)
    tokens = np.array([1, 4, 2])
    vectors = emb.lookup(tokens)
    assert vectors is not None, "lookup() returned None"
    assert vectors.shape == (3, 16), f"Expected shape (3, 16), got {vectors.shape}"
    
    batch_tokens = np.array([[1, 4], [2, 3]])
    batch_vectors = emb.lookup(batch_tokens)
    assert batch_vectors.shape == (2, 2, 16), f"Expected shape (2, 2, 16), got {batch_vectors.shape}"
    print("test_embedding_table passed!")

if __name__ == '__main__':
    try:
        test_embedding_table()
        print("ALL TESTS PASSED: Sprint 1 Ticket 3")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
