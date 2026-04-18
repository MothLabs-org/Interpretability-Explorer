"""Analysis methods: PCA, cosine similarity, representation geometry."""

import torch
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Dict
from .model import ExplorerState


def layerwise_cosine_similarity(state: ExplorerState) -> np.ndarray:
    """Cosine similarity between clean and corrupt mean activations per layer.
    
    Returns: [n_layers] array of cosine similarities.
    """
    clean_mean = state.clean_activations.mean(dim=0)   # [n_layers, d_model]
    corrupt_mean = state.corrupt_activations.mean(dim=0)

    cos = torch.nn.functional.cosine_similarity(clean_mean, corrupt_mean, dim=-1)
    return cos.cpu().numpy()


def layerwise_diff_norms(state: ExplorerState) -> np.ndarray:
    """L2 norm of difference-in-means at each layer.
    
    Returns: [n_layers] array.
    """
    return state.diff_in_means.norm(dim=-1).cpu().numpy()


def cross_layer_cosine_of_directions(state: ExplorerState) -> np.ndarray:
    """Cosine similarity between diff-in-means directions across all layer pairs.
    
    Returns: [n_layers, n_layers] matrix.
    """
    norms = state.diff_in_means.norm(dim=-1, keepdim=True)
    normed = state.diff_in_means / (norms + 1e-8)
    sim_matrix = (normed @ normed.T).cpu().numpy()
    
    return sim_matrix


def pca_at_layer(state: ExplorerState, layer: int, n_components: int = 10) -> Dict:
    """Non-centered SVD of the per-prompt diff vectors at a given layer.

    Returns dict with:
        - explained_variance_ratio: [n_components]
        - components: [n_components, d_model]
        - projections_clean: [n_prompts, n_components]
        - projections_corrupt: [n_prompts, n_components]
        - n_components: int
    """
    diff_stack = state.clean_activations - state.corrupt_activations
    diffs = diff_stack[:, layer, :].cpu().detach().numpy()  # [N, d]

    n_comp = min(n_components, diffs.shape[0], diffs.shape[1])

    _, S, Vt = np.linalg.svd(diffs, full_matrices=False)
    components = Vt[:n_comp] # [n_comp, d]
    explained_variance_ratio = (S[:n_comp] ** 2) / (S ** 2).sum()

    # Project raw activations onto the diff directions
    clean_np = state.clean_activations[:, layer, :].cpu().detach().numpy()
    corrupt_np = state.corrupt_activations[:, layer, :].cpu().detach().numpy()

    return {
        "explained_variance_ratio": explained_variance_ratio,
        "components": components,
        "projections_clean": clean_np @ components.T,
        "projections_corrupt": corrupt_np @ components.T,
        "n_components": n_comp,
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment between two representation matrices.
    
    X, Y: [n_samples, d], should be centered.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    hsic_xy = np.linalg.norm(X.T @ Y, "fro")**2
    hsic_xx = np.linalg.norm(X.T @ X, "fro")**2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro")**2

    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def cross_layer_cka(state: ExplorerState) -> np.ndarray:
    """CKA between clean activations across all layer pairs.
    
    Returns: [n_layers, n_layers] CKA matrix.
    """
    n = state.n_layers
    clean = state.clean_activations.cpu().numpy()  # [n_prompts, n_layers, d_model]
    cka_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            val = linear_cka(clean[:, i, :], clean[:, j, :])
            cka_matrix[i, j] = val
            cka_matrix[j, i] = val

    return cka_matrix



