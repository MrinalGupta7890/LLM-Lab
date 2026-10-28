import sys
from micro_gpt.capstone_micro_gpt import MicroGPTSystem

def test_emergent_behaviors():
    system = MicroGPTSystem()
    res = system.complete("admin sent a secure token to", max_tokens=2)
    assert res is not None, "complete() returned None"
    assert isinstance(res, str), "complete() must return a decoded string"
    print("test_emergent_behaviors passed!")

if __name__ == '__main__':
    try:
        test_emergent_behaviors()
        print("ALL TESTS PASSED: Sprint 6 Ticket 2")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
