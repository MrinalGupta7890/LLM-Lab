\"\"\"
Capstone Project: Build a (Tiny) LLM from Scratch
Vocabulary size: 20 tokens
Emergent behaviors:
1. Long-range size matching ('the big cat sat on the' -> 'big mat')
2. Selective attention / Distractor resistance ('the red big cat sat on the' -> 'big mat')
3. Context diversity ('the cat and the' -> 'dog')
\"\"\"
import numpy as np
from tiny_llm.tokenizer import SimpleTokenizer
from tiny_llm.gpt import GPT
from tiny_llm.sampler import generate

TINY_VOCAB = [
    \"the\", \"cat\", \"dog\", \"sat\", \"ran\", \"on\", \"mat\", \"house\",
    \"a\", \"big\", \"small\", \"quickly\", \"slowly\", \"and\", \"is\",
    \"red\", \"blue\", \"to\", \"PAD\", \"END\"
]

def get_tiny_training_data():
    \"\"\"
    Generate 64 synthetic training patterns showcasing transformer behaviors.
    \"\"\"
    patterns = [
        # Pattern 1: Size matching (long-range dependency)
        \"the big cat sat on the big mat\",
        \"the small dog ran to the small house\",
        \"the big dog sat on the big mat\",
        \"the small cat ran to the small house\",
        # Pattern 2: Color as distractor
        \"the red big cat sat on the big mat\",
        \"the blue small dog ran to the small house\",
        \"the red small cat ran to the small house\",
        \"the blue big dog sat on the big mat\",
        # Pattern 3: Variety over repetition
        \"the cat and the dog sat on the mat\",
        \"the dog and the cat ran to the house\",
    ]
    return patterns

class TinyGPTSystem:
    def __init__(self):
        self.tokenizer = SimpleTokenizer(vocab=TINY_VOCAB)
        self.model = GPT(
            vocab_size=len(self.tokenizer),
            max_seq_len=16,
            d_model=32,
            num_layers=2,
            num_heads=2,
            d_ff=64
        )

    def train_toy_step(self, sentences: list, epochs: int = 5):
        # TODO: Implement toy training loop
        pass

    def complete(self, prompt: str, max_tokens: int = 5) -> str:
        # TODO: Implement prompt completion
        pass
