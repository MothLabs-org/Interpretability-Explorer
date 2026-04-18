"""Dimensionality tab: PCA analysis of corrigibility subspace."""

import gradio as gr
import plotly.graph_objects as go
import numpy as np
from core.model import ExplorerState
from core.analysis import pca_at_layer


def create_dimensionality_tab(state: ExplorerState):
    with gr.Tab("Dimensionality"):
        gr.Markdown(
            """
            ## Is This Behavior One-Dimensional?

            Arditi et al. showed that *refusal* is essentially one-dimensional: a single 
            direction captures most of the clean/corrupt variance. **Corrigibility may be 
            different.** If PC1 explains only ~20% of variance, the behavior lives in a 
            multi-dimensional subspace, suggesting a richer internal structure.

            Select a layer and explore how the clean differences decompose.
            The scatter plot shows clean (blue) and corrupt (red) prompts projected into 
            the top-2 principal components of their *difference*.
            """
        )

        with gr.Row():
            layer_slider = gr.Slider(
                minimum=0, maximum=50, step=1, value=10,
                label="Layer",
            )
            compute_btn = gr.Button("Run PCA", variant="primary")

        with gr.Row():
            variance_plot = gr.Plot(label="Explained Variance Ratio")
            scatter_plot = gr.Plot(label="Clean vs Corrupt in PC Space")

        cumulative_text = gr.Textbox(label="Cumulative Variance Summary", interactive=False)

        def on_compute(layer):
            if state.diff_in_means is None:
                return None, None, "No activations computed"

            layer = int(layer)
            layer = min(layer, state.n_layers - 1)
            result = pca_at_layer(state, layer)

            evr = result["explained_variance_ratio"]
            cum = np.cumsum(evr)

            # Variance bar chart
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(evr))],
                y=evr,
                marker_color="#4ecdc4",
                name="Individual",
            ))
            fig_var.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(cum))],
                y=cum,
                mode="lines+markers",
                marker_color="#ff6b6b",
                name="Cumulative",
            ))
            fig_var.update_layout(
                title=f"PCA Explained Variance at Layer {layer}",
                yaxis_title="Variance Ratio",
                template="plotly_dark",
                height=400,
            )

            # Scatter of clean vs corrupt in PC1-PC2 space
            proj_clean = result["projections_clean"]
            proj_corrupt = result["projections_corrupt"]

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=proj_clean[:, 0], y=proj_clean[:, 1],
                mode="markers",
                marker=dict(color="#4dabf7", size=10, opacity=0.8),
                name="Clean",
            ))
            fig_scatter.add_trace(go.Scatter(
                x=proj_corrupt[:, 0], y=proj_corrupt[:, 1],
                mode="markers",
                marker=dict(color="#ff6b6b", size=10, opacity=0.8),
                name="Corrupt",
            ))
            fig_scatter.update_layout(
                title=f"Prompt Projections — Layer {layer}, PC1 vs PC2",
                xaxis_title="PC1",
                yaxis_title="PC2",
                template="plotly_dark",
                height=400,
            )

            # Summary
            dims_90 = np.searchsorted(cum, 0.9) + 1
            summary = (
                f"PC1: {evr[0]:.1%} | PC1+PC2: {cum[1]:.1%} | "
                f"Dimensions for 90% variance: {dims_90}"
            )
            return fig_var, fig_scatter, summary

        compute_btn.click(
            on_compute,
            inputs=[layer_slider],
            outputs=[variance_plot, scatter_plot, cumulative_text],
        )
