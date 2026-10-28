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

    # Task 1 (50 pts): Verify cold temperature convergence to argmax
    try:
        from micro_gpt.activations import softmax
        logits = np.array([2.0, 1.0, 0.0])
        cold_probs = softmax(logits, temperature=0.01)
        if cold_probs is None:
            tasks["task1"] = make_task_result(0, 50, "softmax() returned None with temperature scaling")
        elif np.argmax(cold_probs) != 0:
            tasks["task1"] = make_task_result(0, 50, f"Cold temperature should peak at index 0 (argmax), got index {np.argmax(cold_probs)}")
        elif cold_probs[0] <= 0.99:
            tasks["task1"] = make_task_result(0, 50, f"Cold temperature should be >= 0.99 at max logit, got {cold_probs[0]}")
        else:
            tasks["task1"] = make_task_result(50, 50, "Cold temperature scaling successfully converges to argmax distribution")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"Cold temperature evaluation error: {str(e)}")

    # Task 2 (50 pts): Verify hot temperature uniform distribution
    try:
        from micro_gpt.activations import softmax
        logits = np.array([2.0, 1.0, 0.0])
        hot_probs = softmax(logits, temperature=100.0)
        if hot_probs is None:
            tasks["task2"] = make_task_result(0, 50, "softmax() returned None for high temperature")
        elif not np.allclose(hot_probs, [0.333, 0.333, 0.333], atol=0.02):
            tasks["task2"] = make_task_result(0, 50, f"Hot temperature should be near uniform [1/3, 1/3, 1/3], got {hot_probs}")
        else:
            tasks["task2"] = make_task_result(50, 50, "Hot temperature scaling smoothly flattens logits into uniform distribution")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"Hot temperature evaluation error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
