"""
Capstone Project: MicroGPT Architecture & Emergent Behaviors
Vocabulary size: 20 tokens
Emergent behaviors:
1. Long-range context preservation ("admin sent a secure token to" -> "database")
2. Selective attention / Distractor resistance ("client fast read a secure token and" -> ignores "fast")
3. Context diversity ("user and client" -> alternates actors)
"""
import numpy as np
from micro_gpt.tokenizer import SimpleTokenizer
from micro_gpt.gpt import GPT
from micro_gpt.sampler import generate

MICRO_VOCAB = [
    "user", "admin", "client", "sent", "read", "wrote", "a",
    "secure", "fast", "token", "query", "record", "and", "data",
    "to", "server", "cache", "database", "<PAD>", "<EOS>"
]

def get_micro_training_data():
    """
    Generate synthetic training patterns showcasing transformer behaviors.
    """
    patterns = [
        # Pattern 1: Long-range context dependency (attribute-entity binding)
        "admin sent a secure token to database",
        "client sent a fast query to cache",
        "user sent a secure record to database",
        "admin wrote a fast token to server",
        # Pattern 2: Modifier as distractor (selective attention)
        "client fast read a secure token to database",
        "user secure read a fast query to cache",
        "admin fast wrote a secure data to server",
        "client secure sent a fast record to cache",
        # Pattern 3: Diversity over repetition (coordination tracking)
        "user and client read data from server",
        "admin and user wrote record to database",
    ]
    return patterns

class MicroGPTSystem:
    def __init__(self):
        self.tokenizer = SimpleTokenizer(vocab=MICRO_VOCAB)
        self.model = GPT(
            vocab_size=len(self.tokenizer),
            max_seq_len=16,
            d_model=32,
            num_layers=2,
            num_heads=2,
            d_ff=64
        )

    def train_toy_step(self, sentences: list, epochs: int = 5):
        # TODO: Implement mini-batch training step
        pass

    def complete(self, prompt: str, max_tokens: int = 5) -> str:
        # TODO: Implement sequence completion rollout
        pass