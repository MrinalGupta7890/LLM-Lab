import sys
from micro_gpt.capstone_micro_gpt import MicroGPTSystem, get_micro_training_data

def test_micro_gpt_training():
    system = MicroGPTSystem()
    data = get_micro_training_data()
    assert len(data) >= 8, "Expected at least 8 training patterns"
    assert len(system.tokenizer) >= 20, "Tokenizer vocabulary must have all 20 tokens"
    print("test_micro_gpt_training passed!")

if __name__ == '__main__':
    try:
        test_micro_gpt_training()
        print("ALL TESTS PASSED: Sprint 6 Ticket 1")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
