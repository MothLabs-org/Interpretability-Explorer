"""Attribution methods: Direct Logit Attribution, logit lens, and CoT reasoning traces."""

import torch
import numpy as np
from typing import Dict, List, Tuple
from .model import ExplorerState


def direct_logit_attribution(
        state: ExplorerState,
        prompt: str,
        target_token: str,
        compare_token: str,
    ) -> Dict:
        target_id = state.model.to_single_token(target_token)
        compare_id = state.model.to_single_token(compare_token)

        with torch.no_grad():
            logits, cache = state.model.run_with_cache(
                prompt,
                names_filter=lambda n: "hook_resid_post" in n or "hook_resid_pre" in n,
            )

            unembed_dir = state.model.W_U[:, target_id] - state.model.W_U[:, compare_id]

            contributions = []
            for layer in range(state.n_layers):
                delta = (
                    cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
                    - cache[f"blocks.{layer}.hook_resid_pre"][0, -1, :]
                )
                contributions.append((delta @ unembed_dir).item())

            actual_diff = (logits[0, -1, target_id] - logits[0, -1, compare_id]).item()
            del cache

        return {
            "layer_contributions": np.array(contributions),
            "total_logit_diff": sum(contributions),
            "actual_logit_diff": actual_diff,
        }

def logit_lens(state: ExplorerState, prompt: str, top_k: int = 5) -> Dict:
    """Apply the logit lens: unembed each layer's residual stream to get token predictions.
    
    Returns dict with:
        - predictions: list of (layer, [(token_str, prob), ...]) for top_k tokens
        - entropies: [n_layers] array of prediction entropy at each layer
    """
    with torch.no_grad():
        _, cache = state.model.run_with_cache(
            prompt,
            names_filter=lambda name: "hook_resid_post" in name,
        )

        ln_final = state.model.ln_final
        W_U = state.model.W_U
        b_U = state.model.b_U 

        predictions = []
        entropies = []

        for layer in range(state.n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]  
            normed = ln_final(resid)
            logits = normed @ W_U + b_U

            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
            entropies.append(entropy)

            top_probs, top_ids = probs.topk(top_k)
            layer_preds = [
                (state.model.tokenizer.decode([tid.item()]), tp.item())
                for tid, tp in zip(top_ids, top_probs)
            ]
            predictions.append((layer, layer_preds))

        del cache
    return {
        "predictions": predictions,
        "entropies": np.array(entropies),
    }


def cot_reasoning_trace(
    state: ExplorerState,
    prompt: str,
    layer: int,
    max_new_tokens: int = 100,
) -> Dict:
    """Generate a CoT response and track the projection onto the clean
    direction at each generated token.
    
    Returns dict with:
        - tokens: list of generated token strings
        - projections: [n_tokens] array of projections onto clean direction
        - generated_text: full generated string
    """
    direction = state.diff_in_means[layer]
    direction = direction / direction.norm()

    input_ids = state.model.to_tokens(prompt)
    generated = input_ids.clone()
    token_strs = []
    projections = []

    hook_name = f"blocks.{layer}.hook_resid_post"

    with torch.no_grad():
        for step in range(max_new_tokens):
            # Capture the activation at the target layer for the last token
            captured = {}

            def capture_hook(activation, hook):
                captured["act"] = activation[0, -1, :].detach()
                return activation

            logits = state.model.run_with_hooks(
                generated,
                fwd_hooks=[(hook_name, capture_hook)],
            )

            # Project onto clean direction
            proj = (captured["act"] @ direction).item()
            next_token = logits[0, -1, :].argmax(dim=-1)

            if next_token.item() in state.model.tokenizer.all_special_ids: #get rid of end of sentence tokens
                break                         

            token_str = state.model.tokenizer.decode([next_token.item()])
            token_strs.append(token_str)
            projections.append(proj)

            generated = torch.cat([generated, next_token.view(1, 1)], dim=-1)

    new_token_ids = generated[0, input_ids.shape[1]:]
    generated_text = state.model.tokenizer.decode(new_token_ids, skip_special_tokens=True)

    return {
        "tokens": token_strs,
        "projections": np.array(projections),
        "generated_text": generated_text,
    }

