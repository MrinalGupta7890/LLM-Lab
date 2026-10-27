import sys
import numpy as np
from tiny_llm.gpt import GPT

def test_gpt_forward():
    model = GPT(vocab_size=20, max_seq_len=16, d_model=32, num_layers=2, num_heads=2, d_ff=64)
    tokens = np.array([[1, 5, 2, 8], [3, 4, 10, 11]])
    logits = model(tokens)
    assert logits is not None, "GPT forward returned None"
    assert logits.shape == (2, 4, 20), f"Expected logits shape (2, 4, 20), got {logits.shape}"
    print("test_gpt_forward passed!")

if __name__ == '__main__':
    try:
        test_gpt_forward()
        print("ALL TESTS PASSED: Sprint 4 Ticket 3")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
