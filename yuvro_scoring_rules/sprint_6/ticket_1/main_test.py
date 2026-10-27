import sys
from tiny_llm.capstone_tiny_gpt import TinyGPTSystem, get_tiny_training_data

def test_tiny_gpt_training():
    system = TinyGPTSystem()
    data = get_tiny_training_data()
    assert len(data) >= 8, "Expected at least 8 training patterns"
    assert len(system.tokenizer) >= 20, "Tokenizer vocabulary must have all 20 tokens"
    print("test_tiny_gpt_training passed!")

if __name__ == '__main__':
    try:
        test_tiny_gpt_training()
        print("ALL TESTS PASSED: Sprint 6 Ticket 1")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
