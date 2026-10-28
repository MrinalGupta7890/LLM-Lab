import sys
import numpy as np
from micro_gpt.gpt import GPT
from micro_gpt.sampler import generate

def test_autoregressive_generation():
    model = GPT(vocab_size=20, max_seq_len=16, d_model=16, num_layers=1, num_heads=2, d_ff=32)
    prompt = [1, 2]
    out_tokens = generate(model, prompt, max_new_tokens=4, temperature=0.7)
    assert out_tokens is not None, "generate() returned None"
    assert len(out_tokens) == 6, f"Expected 6 tokens, got {len(out_tokens)}"
    assert out_tokens[:2] == prompt, "Prompt tokens corrupted"
    print("test_autoregressive_generation passed!")

if __name__ == '__main__':
    try:
        test_autoregressive_generation()
        print("ALL TESTS PASSED: Sprint 5 Ticket 2")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
