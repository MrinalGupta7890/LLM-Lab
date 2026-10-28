from typing import List, Dict, Tuple

class SimpleTokenizer:
    """
    A vocabulary-based subword and word tokenizer with special token support.
    """
    def __init__(self, vocab: List[str] = None):
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        if vocab is None:
            vocab = []
        
        # Ensure special tokens are always present
        full_vocab = list(dict.fromkeys(self.special_tokens + vocab))
        self.token2id: Dict[str, int] = {token: idx for idx, token in enumerate(full_vocab)}
        self.id2token: Dict[int, str] = {idx: token for token, idx in self.token2id.items()}

    def __len__(self) -> int:
        return len(self.token2id)

    def encode(self, text: str) -> List[int]:
        """
        Convert a space-separated string into a list of token IDs.
        Out-of-vocabulary words are mapped to <UNK>.
        """
        # TODO: Implement encode
        pass

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back into a space-separated string.
        Special tokens <PAD>, <BOS>, <EOS> should be omitted from output if present.
        """
        # TODO: Implement decode
        pass

    @staticmethod
    def get_stats(vocab_splits: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Count frequency of adjacent character pairs in vocabulary representations.
        """
        # TODO: Count pair frequencies for BPE
        pass
