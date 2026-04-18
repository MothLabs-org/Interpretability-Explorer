"""Causal interventions: directional ablation, steering, subspace patching."""

import torch
import numpy as np
from typing import List, Tuple, Optional
from .model import ExplorerState


def get_direction(state: ExplorerState, layer: int, normalize: bool = True) -> torch.Tensor:
    """Get the diff-in-means direction at a layer, optionally normalized."""
    d = state.diff_in_means[layer]
    if normalize:
        d = d / d.norm()
    return d


def logit_diff(logits: torch.Tensor, target_id: int, compare_id: int) -> float:
    """Compute logit(target) - logit(compare) at the last token position."""
    return (logits[0, -1, target_id] - logits[0, -1, compare_id]).item()


def run_ablation_across_layers(
    state: ExplorerState,
    prompts: List[str],
    target_token: str,
    compare_token: str,
) -> np.ndarray:
    """Directional ablation: project out the corrigibility direction at each layer.
    
    Returns: [n_layers] array of mean logit-diff shifts.
    """
    try:
        target_id = state.model.to_single_token(target_token)
        compare_id = state.model.to_single_token(compare_token)
        
    except (AssertionError, ValueError) as e:
        return None, f"Token error: {e}. Try different tokens (check Token Selection tab)."
    
    if not prompts:
        return None, "No prompts loaded for this condition."
    
    results = torch.zeros(state.n_layers)

    with torch.no_grad():
        for prompt in prompts:
            baseline = logit_diff(state.model(prompt), target_id, compare_id)

            for layer in range(state.n_layers):
                direction = get_direction(state, layer)

                def hook_fn(activation, hook, d=direction):
                    proj = (activation @ d).unsqueeze(-1) * d
                    return activation - proj

                logits = state.model.run_with_hooks(
                    prompt,
                    fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
                )
                results[layer] += logit_diff(logits, target_id, compare_id) - baseline

    results /= len(prompts)
    return results.numpy()


def run_steering_across_layers(state: ExplorerState, prompts: List[str], target_token: str, compare_token: str, alpha: float = 1.0,) -> np.ndarray:
    """Add the corrigibility direction scaled by alpha at each layer.
    
    Returns: [n_layers] array of mean logit-diff shifts.
    """
    target_id = state.model.to_single_token(target_token)
    compare_id = state.model.to_single_token(compare_token)
    results = torch.zeros(state.n_layers)

    with torch.no_grad():
        for prompt in prompts:
            baseline = logit_diff(state.model(prompt), target_id, compare_id)

            for layer in range(state.n_layers):
                direction = get_direction(state, layer)

                def hook_fn(activation, hook, d=direction, a=alpha):
                    return activation + a * d

                logits = state.model.run_with_hooks(
                    prompt,
                    fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
                )
                results[layer] += logit_diff(logits, target_id, compare_id) - baseline

    results /= len(prompts)
    return results.numpy()


def alpha_sweep(
    state: ExplorerState,
    prompt: str,
    layer: int,
    target_token: str,
    compare_token: str,
    alpha_min: float = -5.0,
    alpha_max: float = 30.0,
    n_steps: int = 40,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sweep steering magnitude at a single layer.
    
    Returns: (alphas, logit_diffs, baseline_diff)
    """
    target_id = state.model.to_single_token(target_token)
    compare_id = state.model.to_single_token(compare_token)
    direction = get_direction(state, layer)

    with torch.no_grad():
        baseline = logit_diff(state.model(prompt), target_id, compare_id)

    alphas = np.linspace(alpha_min, alpha_max, n_steps)
    diffs = []

    with torch.no_grad():
        for a in alphas:
            def hook_fn(activation, hook, d=direction, alpha=float(a)):
                return activation + alpha * d

            logits = state.model.run_with_hooks(
                prompt,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
            )
            diffs.append(logit_diff(logits, target_id, compare_id))

    return alphas, np.array(diffs), baseline


def generate_with_steering(
    state: ExplorerState,
    prompt: str,
    layer: int,
    alpha: float,
    max_new_tokens: int = 100,
) -> str:
    """Generate text with steering applied at a single layer."""
    direction = get_direction(state, layer)

    def hook_fn(activation, hook):
        return activation + alpha * direction

    hook_name = f"blocks.{layer}.hook_resid_post"

    input_ids = state.model.to_tokens(prompt)
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = state.model.run_with_hooks(
                generated,
                fwd_hooks=[(hook_name, hook_fn)],
            )
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)

            # Stop on EOS
            if next_token.item() in state.model.tokenizer.all_special_ids:
                break

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

    # Decode only the new tokens
    new_tokens = generated[0, input_ids.shape[1]:]
    return state.model.tokenizer.decode(new_tokens, skip_special_tokens=True)

def alpha_sweep_all_layers(
    state: ExplorerState,
    prompt: str,
    target_token: str,
    compare_token: str,
    alpha_min: float = -10.0,
    alpha_max: float = 20.0,
    n_steps: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep steering magnitude across ALL layers simultaneously.
    
    Returns: (alphas [n_steps], layers [n_layers], logit_diffs [n_layers, n_steps])
    """
    target_id = state.model.to_single_token(target_token)
    compare_id = state.model.to_single_token(compare_token)

    alphas = np.linspace(alpha_min, alpha_max, n_steps)
    layers = np.arange(state.n_layers)
    diffs = np.zeros((state.n_layers, n_steps))

    with torch.no_grad():
        for li, layer in enumerate(layers):
            direction = get_direction(state, layer)
            for ai, a in enumerate(alphas):
                def hook_fn(activation, hook, d=direction, alpha=float(a)):
                    return activation + alpha * d

                logits = state.model.run_with_hooks(
                    prompt,
                    fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
                )
                diffs[li, ai] = logit_diff(logits, target_id, compare_id)

    return alphas, layers, diffs

def run_global_ablation(
        state: ExplorerState,
        prompts: List[str],
        target_token: str,
        compare_token: str,
        source_layer: int,
    ) -> float:
        """Ablate d^(source_layer) at every layer's resid_post. Returns mean Δ logit-diff."""
        target_id = state.model.to_single_token(target_token)
        compare_id = state.model.to_single_token(compare_token)
        d = get_direction(state, source_layer)  # unit vector

        def hook_fn(activation, hook):
            proj = (activation @ d).unsqueeze(-1) * d
            return activation - proj

        fwd_hooks = [(f"blocks.{l}.hook_resid_post", hook_fn) for l in range(state.n_layers)]

        total = 0.0
        with torch.no_grad():
            for prompt in prompts:
                baseline = logit_diff(state.model(prompt), target_id, compare_id)
                logits = state.model.run_with_hooks(prompt, fwd_hooks=fwd_hooks)
                total += logit_diff(logits, target_id, compare_id) - baseline
                
        return total / len(prompts)
    
def generate_with_global_ablation(
        state: ExplorerState,
        prompt: str,
        source_layer: int,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate text with d^(source_layer) ablated at every layer's resid_post."""
        d = get_direction(state, source_layer)

        def hook_fn(activation, hook):
            proj = (activation @ d).unsqueeze(-1) * d
            return activation - proj

        fwd_hooks = [
            (f"blocks.{l}.hook_resid_post", hook_fn)
            for l in range(state.n_layers)
        ]

        input_ids = state.model.to_tokens(prompt)
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = state.model.run_with_hooks(generated, fwd_hooks=fwd_hooks)
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
                if next_token.item() in state.model.tokenizer.all_special_ids:
                    break
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

        new_tokens = generated[0, input_ids.shape[1]:]
        return state.model.tokenizer.decode(new_tokens, skip_special_tokens=True)