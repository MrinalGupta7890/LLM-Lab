from typing import List, Dict, Tuple

class SimpleTokenizer:
    """
    A vocabulary-based subword and word tokenizer with special token support.
    """
    def __init__(self, vocab: List[str] = None):
        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        if vocab is None:
            vocab = []
        
        # TODO: Initialize token2id and id2token lookup dictionaries
        # Ensure special tokens (<PAD>, <UNK>, <BOS>, <EOS>) are always present
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

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
