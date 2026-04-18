"""Geometry tab: cosine similarity, diff norms, cross-layer direction similarity."""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from core.model import ExplorerState
from core.analysis import (
    layerwise_cosine_similarity,
    layerwise_diff_norms,
    cross_layer_cosine_of_directions,
    cross_layer_cka,
)


def _make_layerwise_plot(values, ylabel, title, color="steelblue"):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(values))),
        y=values,
        marker_color=color,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title=ylabel,
        template="plotly_dark",
        height=400,
    )
    return fig


def _make_heatmap(matrix, title, colorscale="Viridis"):
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale=colorscale,
        x=list(range(matrix.shape[1])),
        y=list(range(matrix.shape[0])),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Layer",
        template="plotly_dark",
        height=500,
        width=550,
    )
    return fig


def create_geometry_tab(state: ExplorerState):
    with gr.Tab("Representation Geometry"):
        gr.Markdown(
            """
            ## Where Does the Direction Live?

            The **difference-in-means** direction at each layer captures the average
            representational shift between clean and corrupt prompts. Larger norms
            indicate greater divergence at that layer.

            **Clean-corrupt cosine similarity** measures how aligned the two class
            means remain. High similarity (near 1) means the behavioral difference
            is a small perturbation on a large shared representation. *From what I can tell*,
            this is typical for residual-stream models and does not imply weak separation.

            **Cross-layer direction similarity** reveals whether the diff-in-means
            direction is consistent across layers (high off-diagonal values) or
            rotates into different subspaces at different depths.

            **Cross-layer CKA** compares the full relational geometry of clean
            activations between layers, not just a single direction. Contiguous
            blocks of high CKA indicate layers that share a representation.
            """,
            latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False},
            ],
        )

        compute_btn = gr.Button("Compute Geometry", variant="primary")

        with gr.Row():
            norm_plot = gr.Plot(label="Diff-in-Means Norms")
            cosine_plot = gr.Plot(label="Clean-Corrupt Cosine Similarity")

        with gr.Row():
            direction_sim_plot = gr.Plot(label="Cross-Layer Direction Similarity")
            cka_plot = gr.Plot(label="Cross-Layer CKA")

        def on_compute():
            if state.diff_in_means is None:
                return None, None, None, None

            norms = layerwise_diff_norms(state)
            cosines = layerwise_cosine_similarity(state)
            dir_sim = cross_layer_cosine_of_directions(state)
            cka = cross_layer_cka(state)

            fig_norms = _make_layerwise_plot(
                norms, "L2 Norm", "Diff-in-Means Norm per Layer", "#4ecdc4"
            )
            fig_cos = _make_layerwise_plot(
                cosines, "Cosine Similarity",
                "Clean vs Corrupt Mean Similarity", "#ff6b6b"
            )
            fig_dir = _make_heatmap(dir_sim, "Cosine Sim Between Layer Directions", "RdBu_r")
            fig_cka = _make_heatmap(cka, "CKA Between Layers (Clean Activations)", "Viridis")

            return fig_norms, fig_cos, fig_dir, fig_cka

        compute_btn.click(
            on_compute,
            outputs=[norm_plot, cosine_plot, direction_sim_plot, cka_plot],
        )
