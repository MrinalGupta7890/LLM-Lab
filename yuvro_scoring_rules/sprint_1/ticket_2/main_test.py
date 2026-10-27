import sys
from tiny_llm.tokenizer import SimpleTokenizer

def test_tokenizer():
    tok = SimpleTokenizer(vocab=["hello", "world", "transformer"])
    assert len(tok) >= 7, "Vocabulary size should include special tokens and words"
    
    encoded = tok.encode("hello transformer world")
    assert encoded is not None, "encode() returned None"
    assert len(encoded) == 3, f"Expected 3 tokens, got {len(encoded)}"
    
    decoded = tok.decode(encoded)
    assert decoded == "hello transformer world", f"Decoded text mismatch: {decoded}"
    
    unk_encoded = tok.encode("unknownword")
    assert unk_encoded[0] == tok.token2id["<UNK>"], "Out-of-vocabulary word should map to <UNK>"
    print("test_tokenizer passed!")

if __name__ == '__main__':
    try:
        test_tokenizer()
        print("ALL TESTS PASSED: Sprint 1 Ticket 2")
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
