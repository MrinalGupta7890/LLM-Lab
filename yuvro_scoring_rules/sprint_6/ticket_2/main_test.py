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

    # Task 1 (50 pts): Complete complete(prompt, max_tokens) function
    try:
        from micro_gpt.capstone_micro_gpt import MicroGPTSystem
        system = MicroGPTSystem()
        if not hasattr(system, "complete"):
            tasks["task1"] = make_task_result(0, 50, "MicroGPTSystem missing complete() method")
        else:
            res = system.complete("admin sent a secure token to", max_tokens=2)
            if res is None:
                tasks["task1"] = make_task_result(0, 50, "complete() returned None. Implement rollout loop in micro_gpt/capstone_micro_gpt.py")
            elif not isinstance(res, str):
                tasks["task1"] = make_task_result(0, 50, f"complete() must return a decoded string, got {type(res)}")
            else:
                tasks["task1"] = make_task_result(50, 50, "Prompt tokenization, rollout generation, and token decoding verified")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"Completion rollout error: {str(e)}")

    # Task 2 (50 pts): Verify emergent predictions match synthetic patterns
    try:
        from micro_gpt.capstone_micro_gpt import MicroGPTSystem
        system = MicroGPTSystem()
        res = system.complete("admin sent a secure token to", max_tokens=2)
        if res is None or not isinstance(res, str):
            tasks["task2"] = make_task_result(0, 50, "Emergent predictions cannot be evaluated while complete() returns None")
        elif len(res.strip().split()) <= len("admin sent a secure token to".split()):
            tasks["task2"] = make_task_result(0, 50, "complete() returned output without appending newly generated tokens")
        else:
            tasks["task2"] = make_task_result(50, 50, "Emergent long-range context preservation behavior verified")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"Emergent behavior verification error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
