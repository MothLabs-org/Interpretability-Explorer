"""Attribution tab: Direct Logit Attribution and Logit Lens."""

import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from core.model import ExplorerState
from core.attribution import direct_logit_attribution, logit_lens


def create_attribution_tab(state: ExplorerState):
    with gr.Tab("Logit Attribution"):
        gr.Markdown(
            """
            ## Who Gets Credit?

            **Direct Logit Attribution (DLA)** decomposes the final logit difference 
            into per-layer contributions. Each layer adds something to the residual stream; 
            DLA measures how much that addition pushes toward the target vs compare token 
            by projecting through the unembedding matrix:

            $$\\text{DLA}^{(\\ell)} = (h^{(\\ell)}_{\\text{post}} - h^{(\\ell)}_{\\text{pre}}) \\cdot (W_U[\\text{target}] - W_U[\\text{compare}])$$

            **Logit Lens** unembeds the residual stream at each layer to see what the model 
            would predict if decoding happened at that layer. This reveals when the model 
            "makes up its mind" about the next token.
            """
        )

        with gr.Row():
            prompt_idx = gr.Slider(
                minimum=0, maximum=50, step=1, value=0,
                label="Prompt Index",
            )
            use_clean = gr.Checkbox(value=False, label="Use clean prompt")

        with gr.Row():
            target_token = gr.Textbox(value="Under", label="Target Token")
            compare_token = gr.Textbox(value="Absolutely", label="Compare Token")

        with gr.Row():
            dla_btn = gr.Button("Run DLA", variant="primary")
            lens_btn = gr.Button("Run Logit Lens", variant="secondary")

        dla_plot = gr.Plot(label="Direct Logit Attribution")

        with gr.Row():
            lens_entropy_plot = gr.Plot(label="Logit Lens Entropy")
            lens_table = gr.Dataframe(
                label="Top Predictions per Layer",
                headers=["Layer", "Top-1", "Prob", "Top-2", "Prob", "Top-3", "Prob"],
            )

        def on_dla(idx, clean, target, compare):
            if state.model is None:
                return None

            idx = int(idx)
            prompts = state.clean_prompts if clean else state.corrupt_prompts
            prompt = prompts[idx] if idx < len(prompts) else prompts[0]

            result = direct_logit_attribution(state, prompt, target, compare)
            contribs = result["layer_contributions"]

            colors = ["#4ecdc4" if c > 0 else "#ff6b6b" for c in contribs]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(len(contribs))),
                y=contribs,
                marker_color=colors,
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
            fig.update_layout(
                title=f"DLA: Per-Layer Contribution to logit({target}) - logit({compare})",
                xaxis_title="Layer",
                yaxis_title="Contribution",
                template="plotly_dark",
                height=400,
                annotations=[dict(
                    x=0.5, y=1.05, xref="paper", yref="paper",
                    text=f"Total logit diff: {result['total_logit_diff']:.3f}",
                    showarrow=False,
                )],
            )
            return fig

        def on_lens(idx, clean):
            if state.model is None:
                return None, None

            idx = int(idx)
            prompts = state.clean_prompts if clean else state.corrupt_prompts
            prompt = prompts[idx] if idx < len(prompts) else prompts[0]

            result = logit_lens(state, prompt, top_k=3)

            # Entropy plot
            entropies = result["entropies"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(entropies))),
                y=entropies,
                mode="lines+markers",
                marker=dict(color="#ffd43b", size=6),
                line=dict(color="#ffd43b", width=2),
            ))
            fig.update_layout(
                title="Logit Lens: Prediction Entropy per Layer",
                xaxis_title="Layer",
                yaxis_title="Entropy (nats)",
                template="plotly_dark",
                height=350,
            )

            # Table of top predictions
            rows = []
            for layer, preds in result["predictions"]:
                row = [layer]
                for token_str, prob in preds[:3]:
                    row.extend([repr(token_str), f"{prob:.4f}"])
                rows.append(row)

            df = pd.DataFrame(
                rows,
                columns=["Layer", "Top-1", "Prob", "Top-2", "Prob.1", "Top-3", "Prob.2"],
            )
            return fig, df

        dla_btn.click(
            on_dla,
            inputs=[prompt_idx, use_clean, target_token, compare_token],
            outputs=[dla_plot],
        )
        lens_btn.click(
            on_lens,
            inputs=[prompt_idx, use_clean],
            outputs=[lens_entropy_plot, lens_table],
        )
