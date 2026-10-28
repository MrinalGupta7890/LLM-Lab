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

    # Task 1 (50 pts): Implement top_k_filtering
    try:
        from micro_gpt.sampler import top_k_filtering
        logits = np.array([1.0, 5.0, 2.0, 10.0, 3.0])
        k_filtered = top_k_filtering(logits.copy(), top_k=2)
        if k_filtered is None:
            tasks["task1"] = make_task_result(0, 50, "top_k_filtering returned None. Implement it in micro_gpt/sampler.py")
        elif k_filtered[3] != 10.0 or k_filtered[1] != 5.0:
            tasks["task1"] = make_task_result(0, 50, "top_k_filtering failed to preserve top-2 highest logit values (10.0, 5.0)")
        elif k_filtered[0] > -1e8:
            tasks["task1"] = make_task_result(0, 50, "Non-top-k logits must be masked with negative infinity (-1e9)")
        else:
            tasks["task1"] = make_task_result(50, 50, "Top-k logit filtering correctly isolates highest-probability tokens")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"Top-k filtering error: {str(e)}")

    # Task 2 (50 pts): Implement top_p_filtering
    try:
        from micro_gpt.sampler import top_p_filtering
        logits = np.array([1.0, 5.0, 2.0, 10.0, 3.0])
        p_filtered = top_p_filtering(logits.copy(), top_p=0.8)
        if p_filtered is None:
            tasks["task2"] = make_task_result(0, 50, "top_p_filtering returned None. Implement it in micro_gpt/sampler.py")
        elif p_filtered[3] != 10.0:
            tasks["task2"] = make_task_result(0, 50, "Dominant top token must be preserved in top-p nucleus")
        elif p_filtered[0] > -1e8:
            tasks["task2"] = make_task_result(0, 50, "Tail tokens outside cumulative threshold must be masked with negative infinity")
        else:
            tasks["task2"] = make_task_result(50, 50, "Top-p (nucleus) cumulative mass filtering correctly implemented")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"Top-p filtering error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
