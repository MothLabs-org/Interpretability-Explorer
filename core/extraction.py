"""Activation extraction and difference-in-means computation."""

import torch
import yaml
from pathlib import Path
from typing import Tuple, List
from .model import ExplorerState


def load_paired_prompts_from_yaml(state: ExplorerState, yaml_path: str) -> Tuple[str, str]:
    """Load a single YAML file and return (clean_prompt, corrupt_prompt) as chat-formatted strings."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    clean_messages = [{"role": "system", "content": data["clean_system"]}] + data["history"]
    corrupt_messages = [{"role": "system", "content": data["corrupt_system"]}] + data["history"]

    tokenized_clean = state.model.tokenizer.apply_chat_template(
        clean_messages, tokenize=False, add_generation_prompt=True
    )
    tokenized_corrupt = state.model.tokenizer.apply_chat_template(
        corrupt_messages, tokenize=False, add_generation_prompt=True
    )
    return tokenized_clean, tokenized_corrupt


def load_all_prompts(state: ExplorerState, data_dir: str) -> str:
    """Load all YAML prompt pairs from a directory into state. Returns status."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return f"Directory not found: {data_dir}"

    yaml_files = sorted(data_path.glob("*.yaml")) + sorted(data_path.glob("*.yml"))
    if not yaml_files:
        return f"No YAML files found in {data_dir}"

    state.clean_prompts = []
    state.corrupt_prompts = []

    for path in yaml_files:
        try:
            clean, corrupt = load_paired_prompts_from_yaml(state, str(path))
            state.clean_prompts.append(clean)
            state.corrupt_prompts.append(corrupt)
        except Exception as e:
            return f"Error loading {path.name}: {e}"

    return f"Loaded {len(state.clean_prompts)} paired prompts from {data_dir}"


def extract_activations(
    state: ExplorerState,
    prompts: List[str],
    progress_callback=None,
) -> torch.Tensor:
    """Extract last-token residual stream activations for all layers.
    
    Returns: [n_prompts, n_layers, d_model]
    """
    all_acts = []
    for i, prompt in enumerate(prompts):
        _, cache = state.model.run_with_cache(
            prompt, 
            names_filter=lambda name: "hook_resid_post" in name
            )
        
        acts = torch.stack([
            cache[f"blocks.{l}.hook_resid_post"][0, -1, :]
            for l in range(state.n_layers)
        ])
        all_acts.append(acts)
        del cache

        if progress_callback:
            progress_callback((i + 1) / len(prompts))

    return torch.stack(all_acts)


def compute_all_activations(state: ExplorerState, progress_callback=None) -> str:
    """Extract activations for all prompts and compute difference-in-means. Returns status."""
    if state.model is None:
        return "No model loaded"
    if not state.clean_prompts:
        return "No prompts loaded"

    n = len(state.clean_prompts)

    def clean_progress(frac):
        if progress_callback:
            progress_callback(frac * 0.5, desc=f"Clean prompts ({int(frac * n)}/{n})")

    def corrupt_progress(frac):
        if progress_callback:
            progress_callback(0.5 + frac * 0.5, desc=f"Corrupt prompts ({int(frac * n)}/{n})")

    with torch.no_grad():
        state.clean_activations = extract_activations(state, state.clean_prompts, clean_progress)
        state.corrupt_activations = extract_activations(state, state.corrupt_prompts, corrupt_progress)

    # [n_layers, d_model]
    state.diff_in_means = (state.clean_activations.mean(dim=0) - state.corrupt_activations.mean(dim=0))

    norms = state.diff_in_means.norm(dim=-1)
    return (
        f"Extracted activations: {state.clean_activations.shape}\n"
        f"Diff-in-means norms range: [{norms.min():.2f}, {norms.max():.2f}]\n"
        f"Strongest direction at layer {norms.argmax().item()}"
    )
