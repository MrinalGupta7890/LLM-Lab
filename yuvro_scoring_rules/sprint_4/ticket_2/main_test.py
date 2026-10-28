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

    # Task 1 (50 pts): Implement GELU activation approximation
    try:
        from micro_gpt.activations import gelu
        x = np.array([-2.0, 0.0, 2.0])
        g = gelu(x)
        if g is None:
            tasks["task1"] = make_task_result(0, 50, "gelu() returned None. Implement it in micro_gpt/activations.py")
        elif not np.isclose(g[1], 0.0, atol=1e-5):
            tasks["task1"] = make_task_result(0, 50, f"GELU(0.0) should be 0.0, got {g[1]}")
        elif g[0] > 0.0 or g[2] < 1.9:
            tasks["task1"] = make_task_result(0, 50, f"GELU output out of expected range: {g}")
        else:
            tasks["task1"] = make_task_result(50, 50, "Smooth Gaussian Error Linear Unit (GELU) approximation verified")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"GELU activation error: {str(e)}")

    # Task 2 (50 pts): Implement FFN forward projection
    try:
        from micro_gpt.transformer_block import FeedForward
        ffn = FeedForward(d_model=16, d_ff=64)
        inp = np.random.randn(2, 4, 16)
        out = ffn(inp)
        if out is None:
            tasks["task2"] = make_task_result(0, 50, "FeedForward forward() returned None. Implement it in micro_gpt/transformer_block.py")
        elif out.shape != (2, 4, 16):
            tasks["task2"] = make_task_result(0, 50, f"FeedForward output shape mismatch: expected (2, 4, 16), got {out.shape}")
        else:
            tasks["task2"] = make_task_result(50, 50, "Two-stage non-linear feedforward projection (d_model -> d_ff -> d_model) verified")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"FeedForward projection error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
