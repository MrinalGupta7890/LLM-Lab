import os
import sys
import json

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

    # Task 1 (30 pts): Initialize token2id and id2token lookup dictionaries
    try:
        from micro_gpt.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer(vocab=["hello", "world", "transformer"])
        if not hasattr(tok, "token2id") or not hasattr(tok, "id2token"):
            tasks["task1"] = make_task_result(0, 30, "SimpleTokenizer missing token2id or id2token dictionary")
        elif len(tok) < 7:
            tasks["task1"] = make_task_result(0, 30, f"Vocabulary size should include special tokens and words (expected >= 7, got {len(tok)})")
        elif "<UNK>" not in tok.token2id or "<PAD>" not in tok.token2id:
            tasks["task1"] = make_task_result(0, 30, "Special tokens <PAD> or <UNK> missing from vocabulary mapping")
        else:
            tasks["task1"] = make_task_result(30, 30, "Vocabulary and special token dictionaries initialized successfully")
    except Exception as e:
        tasks["task1"] = make_task_result(0, 30, f"Vocabulary initialization error: {str(e)}")

    # Task 2 (40 pts): Implement encode(text) -> List[int]
    try:
        from micro_gpt.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer(vocab=["hello", "world", "transformer"])
        encoded = tok.encode("hello transformer world")
        if encoded is None:
            tasks["task2"] = make_task_result(0, 40, "encode() returned None. Implement it in micro_gpt/tokenizer.py")
        elif len(encoded) != 3:
            tasks["task2"] = make_task_result(0, 40, f"Expected 3 token IDs, got {len(encoded)}")
        else:
            unk_encoded = tok.encode("unknownword")
            if not unk_encoded or unk_encoded[0] != tok.token2id.get("<UNK>"):
                tasks["task2"] = make_task_result(0, 40, "Out-of-vocabulary word should map to <UNK> token ID")
            else:
                tasks["task2"] = make_task_result(40, 40, "encode() maps in-vocab words and handles <UNK> tokens correctly")
    except Exception as e:
        tasks["task2"] = make_task_result(0, 40, f"encode() evaluation error: {str(e)}")

    # Task 3 (30 pts): Implement decode(token_ids) -> str
    try:
        from micro_gpt.tokenizer import SimpleTokenizer
        tok = SimpleTokenizer(vocab=["hello", "world", "transformer"])
        encoded = tok.encode("hello transformer world")
        decoded = tok.decode(encoded) if encoded is not None else None
        if decoded is None:
            tasks["task3"] = make_task_result(0, 30, "decode() returned None. Implement it in micro_gpt/tokenizer.py")
        elif decoded != "hello transformer world":
            tasks["task3"] = make_task_result(0, 30, f"Decoded text mismatch: expected 'hello transformer world', got '{decoded}'")
        else:
            tasks["task3"] = make_task_result(30, 30, "decode() successfully reconstructs text and filters special tokens")
    except Exception as e:
        tasks["task3"] = make_task_result(0, 30, f"decode() evaluation error: {str(e)}")

    print_result(tasks)

if __name__ == '__main__':
    evaluate()
