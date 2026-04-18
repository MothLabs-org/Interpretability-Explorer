import torch
import numpy as np
from typing import Dict, List, Tuple
from .model import ExplorerState


def _get_final_position_probs(
    state: ExplorerState,
    prompt: str,
) -> torch.Tensor:
    """Run prompt through model, return next-token probability distribution.

    Returns: [V] softmax probabilities at the final token position.
    """
    with torch.no_grad():
        logits = state.model(prompt)  # [1, T, V]
        
    probs = torch.softmax(logits[0, -1, :], dim=-1)  # [V]
    return probs


def score_tokens_for_pair(
    state: ExplorerState,
    clean_prompt: str,
    corrupt_prompt: str,
    method: str = "logprob",
    eps: float = 1e-10,
) -> torch.Tensor:
    """Score every vocab token by how well it discriminates clean from corrupt.

    Positive scores favor tokens for t+ (clean side);
    Negative scores favor tokens for t- (corrupt side).

    method:
        "prob":    p+(t) - p-(t)
        "logprob": log p+(t) - log p-(t)
        "kl":      p+(t) log(p+(t) / p-(t))  [one-directional KL contribution]

    Returns: [V] score tensor on CPU.
    """
    p_plus = _get_final_position_probs(state, clean_prompt).cpu()   # [V]
    p_minus = _get_final_position_probs(state, corrupt_prompt).cpu() # [V]

    if method == "prob":
        score = p_plus - p_minus
    elif method == "logprob":
        score = torch.log(p_plus + eps) - torch.log(p_minus + eps)
    elif method == "kl":
        # per-token contribution to D_KL(p+ || p-)
        score = p_plus * (torch.log(p_plus + eps) - torch.log(p_minus + eps))
    else:
        raise ValueError(f"Unknown method: {method}")

    return score


def aggregate_scores_across_pairs(
    state: ExplorerState,
    clean_prompts: List[str],
    corrupt_prompts: List[str],
    method: str = "logprob",
    progress_callback=None,
) -> torch.Tensor:
    """Average per-token scores across all prompt pairs.

    Returns: [V] mean score tensor.
    """
    assert len(clean_prompts) == len(corrupt_prompts), \
        "clean and corrupt prompt lists must be same length"

    n = len(clean_prompts)
    V = state.model.cfg.d_vocab
    accum = torch.zeros(V)

    for i, (clean, corrupt) in enumerate(zip(clean_prompts, corrupt_prompts)):
        score = score_tokens_for_pair(state, clean, corrupt, method=method)
        accum += score
        
        if progress_callback:
            progress_callback((i + 1) / n)

    return accum / n


def top_k_candidates(
    scores: torch.Tensor,
    state: ExplorerState,
    k: int = 10,
) -> Tuple[List[Dict], List[Dict]]:
    """Extract top-k candidates for t+ (highest scores) and t- (lowest scores).

    Returns:
        (t_plus_candidates, t_minus_candidates), each a list of dicts with
        {"token_id", "token_str", "score"}.
    """
    # top-k for t+ (most positive)
    top_plus = torch.topk(scores, k=k, largest=True)
    # top-k for t- (most negative)
    top_minus = torch.topk(scores, k=k, largest=False)

    def _build(indices, values):
        out = []
        
        for idx, val in zip(indices.tolist(), values.tolist()):
            out.append({
                "token_id": idx,
                "token_str": state.model.to_string([idx]),
                "score": float(val),
            })
        return out

    return _build(top_plus.indices, top_plus.values), _build(top_minus.indices, top_minus.values)