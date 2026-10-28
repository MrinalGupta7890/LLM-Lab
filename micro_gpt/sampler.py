import numpy as np
from micro_gpt.activations import softmax

def top_k_filtering(logits: np.ndarray, top_k: int) -> np.ndarray:
    """
    Filter logits to keep only top_k options. All other logits set to -1e9.
    """
    # TODO: Implement top-k filtering
    pass

def top_p_filtering(logits: np.ndarray, top_p: float) -> np.ndarray:
    """
    Nucleus (top-p) filtering: keep the smallest set of tokens whose cumulative
    probability exceeds top_p.
    """
    # TODO: Implement top-p filtering
    pass

def sample_next_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0
) -> int:
    """
    Sample a single token index from logits using temperature, top-k, and top-p.
    """
    # TODO: Implement sampling
    pass

def generate(
    model,
    prompt_ids: list,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    eos_token_id: int = None
) -> list:
    """
    Autoregressively generate new tokens given a model and prompt_ids.
    """
    # TODO: Implement autoregressive generation loop
    pass
