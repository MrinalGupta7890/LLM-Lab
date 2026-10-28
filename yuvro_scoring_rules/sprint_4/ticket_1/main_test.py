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

    # Task 1 (50 pts): Compute mean and variance with keepdims=True
    try:
        from micro_gpt.transformer_block import LayerNorm
        ln = LayerNorm(d_model=8)
        x = np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]])
        out = ln(x)
        if out is None:
            tasks["task1"] = make_task_result(0, 50, "LayerNorm forward() returned None. Implement it in micro_gpt/transformer_block.py")
        elif not np.isclose(np.mean(out), 0.0, atol=1e-4):
            tasks["task1"] = make_task_result(0, 50, f"Standardized output mean should be 0.0, got {np.mean(out)}")
        elif not np.isclose(np.var(out), 1.0, atol=1e-2):
            tasks["task1"] = make_task_result(0, 50, f"Standardized output variance should be 1.0, got {np.var(out)}")
        else:
            tasks["task1"] = make_task_result(50, 50, "Feature standardization (mean=0, variance=1) verified with keepdims")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"Standardization computation error: {str(e)}")

    # Task 2 (50 pts): Apply learnable gamma and beta transformations
    try:
        from micro_gpt.transformer_block import LayerNorm
        ln = LayerNorm(d_model=8)
        if not hasattr(ln, "gamma") or not hasattr(ln, "beta"):
            tasks["task2"] = make_task_result(0, 50, "LayerNorm missing learnable parameters gamma or beta")
        elif not np.allclose(ln.gamma, 1.0) or not np.allclose(ln.beta, 0.0):
            tasks["task2"] = make_task_result(0, 50, "LayerNorm gamma must initialize to 1.0 and beta to 0.0")
        else:
            tasks["task2"] = make_task_result(50, 50, "Affine scale (gamma) and shift (beta) transformations verified")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"Parameter verification error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
