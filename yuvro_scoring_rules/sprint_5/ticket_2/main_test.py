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

    # Task 1 (50 pts): Extract last token logits from forward output
    try:
        from micro_gpt.gpt import GPT
        model = GPT(vocab_size=20, max_seq_len=16, d_model=16, num_layers=1, num_heads=2, d_ff=32)
        tokens = np.array([[1, 2, 3]])
        logits = model(tokens)
        if logits is None:
            tasks["task1"] = make_task_result(0, 50, "GPT model forward pass returned None")
        elif logits.shape != (1, 3, 20):
            tasks["task1"] = make_task_result(0, 50, f"Expected logits shape (1, 3, 20), got {logits.shape}")
        else:
            last_logits = logits[:, -1, :]
            if last_logits.shape != (1, 20):
                tasks["task1"] = make_task_result(0, 50, f"Last token logit slice expected shape (1, 20), got {last_logits.shape}")
            else:
                tasks["task1"] = make_task_result(50, 50, "Last token logits correctly extracted for next-token prediction")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"Next-token logit extraction error: {str(e)}")

    # Task 2 (50 pts): Iteratively append sampled token to context
    try:
        from micro_gpt.gpt import GPT
        from micro_gpt.sampler import generate
        model = GPT(vocab_size=20, max_seq_len=16, d_model=16, num_layers=1, num_heads=2, d_ff=32)
        prompt = [1, 2]
        out_tokens = generate(model, prompt, max_new_tokens=4, temperature=0.7)
        if out_tokens is None:
            tasks["task2"] = make_task_result(0, 50, "generate() returned None. Implement it in micro_gpt/sampler.py")
        elif len(out_tokens) != 6:
            tasks["task2"] = make_task_result(0, 50, f"Expected 6 tokens (2 prompt + 4 generated), got {len(out_tokens)}")
        elif out_tokens[:2] != prompt:
            tasks["task2"] = make_task_result(0, 50, "Prompt tokens were corrupted during autoregressive generation")
        else:
            tasks["task2"] = make_task_result(50, 50, "Autoregressive generation loop correctly rolls out new tokens onto context")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"Autoregressive generation error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
