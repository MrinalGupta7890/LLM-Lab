import os
import sys
import json
import numpy as np

REPO_ROOT = os.getcwd()
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def make_task_result(score, max_score, message):
    passed = score >= max_score and max_score > 0
    return {
        "score": score,
        "max_score": max_score,
        "result": "passed" if passed else "failed",
        "message": message
    }

def print_result(tasks):
    total_score = sum(t["score"] for t in tasks.values())
    max_possible = sum(t["max_score"] for t in tasks.values())
    output = {
        "tasks": tasks,
        "total_score": total_score,
        "max_possible_score": max_possible
    }
    print(json.dumps(output, indent=2))

def evaluate():
    tasks = {}

    # Task 1 (40 pts): Implement Pre-LN TransformerBlock
    try:
        from micro_gpt.transformer_block import TransformerBlock
        block = TransformerBlock(d_model=16, num_heads=2, d_ff=32)
        x = np.random.randn(2, 4, 16)
        block_out = block(x, is_causal=True)
        if block_out is None:
            tasks["task1"] = make_task_result(0, 40, "TransformerBlock forward() returned None. Implement it in micro_gpt/transformer_block.py")
        elif block_out.shape != (2, 4, 16):
            tasks["task1"] = make_task_result(0, 40, f"TransformerBlock output shape mismatch: expected (2, 4, 16), got {block_out.shape}")
        else:
            tasks["task1"] = make_task_result(40, 40, "Pre-LN TransformerBlock with residual connections verified")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 40, f"TransformerBlock error: {str(e)}")

    # Task 2 (60 pts): Implement GPT forward pass with embeddings
    try:
        from micro_gpt.gpt import GPT
        model = GPT(vocab_size=20, max_seq_len=16, d_model=32, num_layers=2, num_heads=2, d_ff=64)
        tokens = np.array([[1, 5, 2, 8], [3, 4, 10, 11]])
        logits = model(tokens)
        if logits is None:
            tasks["task2"] = make_task_result(0, 60, "GPT forward() returned None. Implement it in micro_gpt/gpt.py")
        elif logits.shape != (2, 4, 20):
            tasks["task2"] = make_task_result(0, 60, f"Expected logits shape (2, 4, 20), got {logits.shape}")
        else:
            tasks["task2"] = make_task_result(60, 60, "Complete GPT forward pass (token + pos embeddings, N blocks, final LM head) verified")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 60, f"GPT forward pass error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
