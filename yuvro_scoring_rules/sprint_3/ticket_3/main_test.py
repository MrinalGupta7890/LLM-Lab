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
    batch, seq_len, d_model, num_heads = 2, 4, 16, 4

    # Task 1 (40 pts): Implement Q, K, V projections and head reshaping
    try:
        from micro_gpt.attention import MultiHeadAttention
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        if not hasattr(mha, "W_q") or not hasattr(mha, "W_k") or not hasattr(mha, "W_v"):
            tasks["task1"] = make_task_result(0, 40, "MultiHeadAttention missing linear projection matrices W_q, W_k, W_v")
        elif mha.d_k != d_model // num_heads:
            tasks["task1"] = make_task_result(0, 40, f"Head dimension d_k incorrect: expected {d_model // num_heads}, got {mha.d_k}")
        else:
            tasks["task1"] = make_task_result(40, 40, "Q, K, V projection matrices and subspace head splitting verified")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 40, f"Subspace projection initialization error: {str(e)}")

    # Task 2 (30 pts): Compute parallel scaled dot-product attention
    try:
        from micro_gpt.attention import MultiHeadAttention
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        x = np.random.randn(batch, seq_len, d_model)
        out = mha(x, is_causal=True)
        if out is None:
            tasks["task2"] = make_task_result(0, 30, "MultiHeadAttention forward returned None. Implement it in micro_gpt/attention.py")
        elif not isinstance(out, np.ndarray):
            tasks["task2"] = make_task_result(0, 30, f"Expected numpy array output, got {type(out)}")
        else:
            tasks["task2"] = make_task_result(30, 30, "Parallel causal attention computation successfully executed across all heads")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 30, f"Parallel attention execution error: {str(e)}")

    # Task 3 (30 pts): Concatenate heads and apply W_O
    try:
        from micro_gpt.attention import MultiHeadAttention
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        x = np.random.randn(batch, seq_len, d_model)
        out = mha(x, is_causal=True)
        if out is None:
            tasks["task3"] = make_task_result(0, 30, "MultiHeadAttention output is None")
        elif out.shape != (batch, seq_len, d_model):
            tasks["task3"] = make_task_result(0, 30, f"Output shape mismatch: expected (2, 4, 16), got {out.shape}")
        else:
            tasks["task3"] = make_task_result(30, 30, "Head concatenation and output projection W_O verified matching model dimension")
    except Exception as e:
        tasks["task3"] = make_task_result(0, 30, f"Head concatenation error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
