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

    # Task 1 (50 pts): Implement lookup supporting batches
    try:
        from micro_gpt.embeddings import EmbeddingTable
        emb = EmbeddingTable(vocab_size=10, d_model=16)
        batch_tokens = np.array([[1, 4], [2, 3]])
        batch_vectors = emb.lookup(batch_tokens)
        if batch_vectors is None:
            tasks["task1"] = make_task_result(0, 50, "lookup() returned None. Implement it in micro_gpt/embeddings.py")
        elif not isinstance(batch_vectors, np.ndarray):
            tasks["task1"] = make_task_result(0, 50, f"lookup() must return a numpy array, got {type(batch_vectors)}")
        elif batch_vectors.shape != (2, 2, 16):
            tasks["task1"] = make_task_result(0, 50, f"Batch lookup shape mismatch: expected (2, 2, 16), got {batch_vectors.shape}")
        else:
            tasks["task1"] = make_task_result(50, 50, "EmbeddingTable.lookup successfully handles multi-dimensional batches")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 50, f"Embedding lookup error: {str(e)}")

    # Task 2 (50 pts): Verify output shapes match (batch, seq_len, d_model)
    try:
        from micro_gpt.embeddings import EmbeddingTable
        emb = EmbeddingTable(vocab_size=10, d_model=16)
        tokens = np.array([1, 4, 2])
        vectors = emb.lookup(tokens)
        if vectors is None:
            tasks["task2"] = make_task_result(0, 50, "lookup() returned None for 1D token sequence")
        elif vectors.shape != (3, 16):
            tasks["task2"] = make_task_result(0, 50, f"Expected shape (3, 16), got {vectors.shape}")
        else:
            tasks["task2"] = make_task_result(50, 50, "Output tensor shapes verified matching (..., d_model)")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 50, f"Shape verification error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
