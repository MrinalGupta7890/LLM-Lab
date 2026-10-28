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
    seq_len, d_k, d_v = 4, 8, 8
    q = np.random.randn(1, seq_len, d_k)
    k = np.random.randn(1, seq_len, d_k)
    v = np.random.randn(1, seq_len, d_v)

    # Task 1 (40 pts): Compute scaled dot product scores
    try:
        from micro_gpt.attention import scaled_dot_product_attention
        out, weights = scaled_dot_product_attention(q, k, v)
        if weights is None:
            tasks["task1"] = make_task_result(0, 40, "scaled_dot_product_attention returned None weights")
        elif weights.shape != (1, seq_len, seq_len):
            tasks["task1"] = make_task_result(0, 40, f"Attention scores matrix shape mismatch: expected (1, 4, 4), got {weights.shape}")
        else:
            tasks["task1"] = make_task_result(40, 40, "Scaled dot product score matrix (Q @ K.T / sqrt(d_k)) correctly computed")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 40, f"Dot product score calculation error: {str(e)}")

    # Task 2 (30 pts): Apply softmax to obtain attention weights
    try:
        from micro_gpt.attention import scaled_dot_product_attention
        out, weights = scaled_dot_product_attention(q, k, v)
        if weights is None:
            tasks["task2"] = make_task_result(0, 30, "Attention weights are None")
        elif not np.allclose(np.sum(weights, axis=-1), 1.0, atol=1e-5):
            tasks["task2"] = make_task_result(0, 30, f"Attention weights must sum to 1.0 along sequence dimension: {np.sum(weights, axis=-1)}")
        else:
            tasks["task2"] = make_task_result(30, 30, "Softmax correctly produces normalized attention probability distribution")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 30, f"Attention weights softmax error: {str(e)}")

    # Task 3 (30 pts): Multiply attention weights by Value tensor
    try:
        from micro_gpt.attention import scaled_dot_product_attention
        out, weights = scaled_dot_product_attention(q, k, v)
        if out is None:
            tasks["task3"] = make_task_result(0, 30, "scaled_dot_product_attention output context tensor is None")
        elif out.shape != (1, seq_len, d_v):
            tasks["task3"] = make_task_result(0, 30, f"Context output shape mismatch: expected (1, 4, 8), got {out.shape}")
        else:
            tasks["task3"] = make_task_result(30, 30, "Context output correctly aggregated across Value vectors")
    except Exception as e:
        tasks["task3"] = make_task_result(0, 30, f"Context output multiplication error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
