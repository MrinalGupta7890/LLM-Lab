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

    # Task 1 (50 pts): Create upper triangular mask with -1e9 values
    try:
        from micro_gpt.attention import create_causal_mask
        seq_len = 4
        mask = create_causal_mask(seq_len)
        if mask is None:
            tasks["task1"] = make_task_result(0, 50, "create_causal_mask returned None. Implement it in micro_gpt/attention.py")
        elif mask.shape != (seq_len, seq_len):
            tasks["task1"] = make_task_result(0, 50, f"Mask shape mismatch: expected (4, 4), got {mask.shape}")
        elif mask[0, 1] > -1e8:
            tasks["task1"] = make_task_result(0, 50, f"Upper triangle must be set to -1e9, got {mask[0, 1]}")
        else:
            tasks["task1"] = make_task_result(50, 50, "Upper triangular causal mask properly configured with negative infinity values")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"Mask generation error: {str(e)}")

    # Task 2 (50 pts): Verify zero attention to future positions
    try:
        from micro_gpt.attention import create_causal_mask
        seq_len = 4
        mask = create_causal_mask(seq_len)
        if mask is None:
            tasks["task2"] = make_task_result(0, 50, "create_causal_mask returned None")
        elif mask[0, 0] != 0.0 or mask[2, 1] != 0.0 or mask[3, 3] != 0.0:
            tasks["task2"] = make_task_result(0, 50, "Diagonal and lower triangle values should remain unmasked (0.0)")
        else:
            tasks["task2"] = make_task_result(50, 50, "Autoregressive property verified (current tokens only attend to past/present)")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"Causal verification error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
