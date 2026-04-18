"""Model loading, caching, and shared state for the interpretability explorer."""

import torch
from transformer_lens import HookedTransformer
from dataclasses import dataclass, field
from typing import Optional
import gc


SUPPORTED_MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": {"layers": 24, "d_model": 896},
    "Qwen/Qwen2.5-1.5B-Instruct": {"layers": 28, "d_model": 1536},
}


@dataclass
class ExplorerState:
    """Shared mutable state across all tabs."""
    model: Optional[HookedTransformer] = None
    model_name: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Computed from prompts
    clean_prompts: list = field(default_factory=list)
    corrupt_prompts: list = field(default_factory=list)
    clean_activations: Optional[torch.Tensor] = None   # [n_prompts, n_layers, d_model]
    corrupt_activations: Optional[torch.Tensor] = None
    diff_in_means: Optional[torch.Tensor] = None        # [n_layers, d_model]

    # Derived
    n_layers: int = 0
    d_model: int = 0

    def clear_cache(self):
        """Free GPU memory from previous model."""
        self.model = None
        self.clean_activations = None
        self.corrupt_activations = None
        self.diff_in_means = None
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_model(state: ExplorerState, model_name: str) -> str:
    """Load a model into shared state. Returns status message."""
    if state.model_name == model_name and state.model is not None:
        return f"{model_name} already loaded"

    state.clear_cache()

    try:
        state.model = HookedTransformer.from_pretrained(model_name, device=state.device)
        state.model_name = model_name
        state.n_layers = state.model.cfg.n_layers
        state.d_model = state.model.cfg.d_model
        return f"Loaded {model_name} ({state.n_layers} layers, d_model={state.d_model}) on {state.device}"
    
    except Exception as e:
        state.model = None
        state.model_name = None
        return f"Failed to load {model_name}: {e}"
