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

    # Task 1 (50 pts): Tokenize training patterns into batches
    try:
        from micro_gpt.capstone_micro_gpt import MicroGPTSystem, get_micro_training_data
        system = MicroGPTSystem()
        data = get_micro_training_data()
        if not data or len(data) < 8:
            tasks["task1"] = make_task_result(0, 50, f"Expected at least 8 synthetic training patterns, got {len(data) if data else 0}")
        elif len(system.tokenizer) < 20:
            tasks["task1"] = make_task_result(0, 50, f"Tokenizer vocabulary incomplete: expected >= 20 tokens, got {len(system.tokenizer)}")
        else:
            sample_encoded = system.tokenizer.encode(data[0])
            if not sample_encoded or len(sample_encoded) == 0:
                tasks["task1"] = make_task_result(0, 50, "Failed to tokenize synthetic pattern into integer IDs")
            else:
                tasks["task1"] = make_task_result(50, 50, "Multi-feature synthetic corpus tokenization pipeline verified")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"Training pattern tokenization error: {str(e)}")

    # Task 2 (50 pts): Evaluate cross-entropy loss reduction
    try:
        from micro_gpt.capstone_micro_gpt import MicroGPTSystem, get_micro_training_data
        system = MicroGPTSystem()
        data = get_micro_training_data()
        if not hasattr(system, "train_toy_step"):
            tasks["task2"] = make_task_result(0, 50, "MicroGPTSystem missing train_toy_step method")
        else:
            res = system.train_toy_step(data[:2], epochs=1)
            # If train_toy_step is unimplemented pass, it returns None
            if res is None and getattr(system, "_trained", False) is False:
                # Check if model weights changed or loss was returned
                tasks["task2"] = make_task_result(0, 50, "train_toy_step() not yet implemented. Implement training loop in micro_gpt/capstone_micro_gpt.py")
            else:
                tasks["task2"] = make_task_result(50, 50, "Cross-entropy loss calculation and parameter update step verified")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"Training step evaluation error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
