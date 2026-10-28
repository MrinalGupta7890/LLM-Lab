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

    # Task 1 (50 pts): Implement subtract-max normalization trick
    try:
        from micro_gpt.activations import softmax
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = softmax(logits)
        if probs is None:
            tasks["task1"] = make_task_result(0, 50, "softmax() returned None. Implement it in micro_gpt/activations.py")
        elif np.isnan(probs).any():
            tasks["task1"] = make_task_result(0, 50, "softmax produced NaN on large logits; subtract max(logits) before exponentiating")
        elif np.isinf(probs).any():
            tasks["task1"] = make_task_result(0, 50, "softmax produced Inf on large logits")
        else:
            tasks["task1"] = make_task_result(50, 50, "Numerically stable softmax with subtract-max trick verified")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"Numerical stability error: {str(e)}")

    # Task 2 (50 pts): Ensure probabilities sum to 1 along target axis
    try:
        from micro_gpt.activations import softmax
        batch_logits = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        batch_probs = softmax(batch_logits, axis=-1)
        if batch_probs is None:
            tasks["task2"] = make_task_result(0, 50, "softmax() returned None for 2D batch")
        elif not np.allclose(np.sum(batch_probs, axis=-1), [1.0, 1.0], atol=1e-5):
            tasks["task2"] = make_task_result(0, 50, f"Batch probabilities along axis do not sum to 1.0: {np.sum(batch_probs, axis=-1)}")
        else:
            tasks["task2"] = make_task_result(50, 50, "Probabilities correctly normalized to 1.0 along target axis")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"Probability normalization error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
