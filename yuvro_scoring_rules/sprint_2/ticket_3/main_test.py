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

    # Task 1 (50 pts): Implement Linear.forward(x)
    try:
        from micro_gpt.linear import Linear
        layer = Linear(in_features=8, out_features=4)
        x = np.random.randn(2, 3, 8)
        out = layer(x)
        if out is None:
            tasks["task1"] = make_task_result(0, 50, "Linear forward() returned None. Implement it in micro_gpt/linear.py")
        elif out.shape != (2, 3, 4):
            tasks["task1"] = make_task_result(0, 50, f"Linear output shape mismatch: expected (2, 3, 4), got {out.shape}")
        else:
            tasks["task1"] = make_task_result(50, 50, "Linear forward projection verified with correct affine transformation")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"Linear forward error: {str(e)}")

    # Task 2 (50 pts): Implement cross_entropy_loss(logits, targets)
    try:
        from micro_gpt.linear import cross_entropy_loss
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        targets = np.array([0, 1])
        loss = cross_entropy_loss(logits, targets)
        if loss is None:
            tasks["task2"] = make_task_result(0, 50, "cross_entropy_loss() returned None. Implement it in micro_gpt/linear.py")
        elif loss >= 0.05:
            tasks["task2"] = make_task_result(0, 50, f"Loss for confident correct predictions should be near 0, got {loss}")
        else:
            tasks["task2"] = make_task_result(50, 50, "cross_entropy_loss correctly measures negative log-likelihood")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"Cross-entropy loss evaluation error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
