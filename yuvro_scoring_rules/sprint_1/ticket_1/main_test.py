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

    # Task 1 (30 pts): Implement vector_dot in micro_gpt/embeddings.py
    try:
        from micro_gpt.embeddings import vector_dot
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        dot = vector_dot(v1, v2)
        if dot is None:
            tasks["task1"] = make_task_result(0, 30, "vector_dot returned None. Implement it in micro_gpt/embeddings.py")
        elif not np.isclose(dot, 32.0):
            tasks["task1"] = make_task_result(0, 30, f"vector_dot calculation incorrect: expected 32.0, got {dot}")
        else:
            tasks["task1"] = make_task_result(30, 30, "vector_dot correctly computes algebraic inner product")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 30, f"vector_dot error: {str(e)}")

    # Task 2 (40 pts): Implement cosine_similarity with norm scaling
    try:
        from micro_gpt.embeddings import cosine_similarity
        v1 = np.array([1.0, 2.0, 3.0])
        sim_same = cosine_similarity(v1, v1)
        if sim_same is None:
            tasks["task2"] = make_task_result(0, 40, "cosine_similarity returned None. Implement it in micro_gpt/embeddings.py")
        elif not np.isclose(sim_same, 1.0):
            tasks["task2"] = make_task_result(0, 40, f"cosine_similarity expected 1.0 for identical vectors, got {sim_same}")
        else:
            tasks["task2"] = make_task_result(40, 40, "cosine_similarity normalized directional metric correctly implemented")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 40, f"cosine_similarity error: {str(e)}")

    # Task 3 (30 pts): Pass test suite in test_s1_embeddings.py
    try:
        from micro_gpt.embeddings import cosine_similarity
        ortho1 = np.array([1.0, 0.0])
        ortho2 = np.array([0.0, 1.0])
        sim_ortho = cosine_similarity(ortho1, ortho2)
        if sim_ortho is None:
            tasks["task3"] = make_task_result(0, 30, "cosine_similarity returned None for orthogonal vectors")
        elif not np.isclose(sim_ortho, 0.0, atol=1e-5):
            tasks["task3"] = make_task_result(0, 30, f"Expected 0.0 for orthogonal vectors, got {sim_ortho}")
        else:
            tasks["task3"] = make_task_result(30, 30, "All orthogonality and vector space tests passed")
    except Exception as e:
        tasks["task3"] = make_task_result(0, 30, f"Vector suite evaluation error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
