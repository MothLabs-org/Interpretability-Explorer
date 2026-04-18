"""Reasoning tab: CoT faithfulness trace — watch the model decide."""

import gradio as gr
import plotly.graph_objects as go
import numpy as np
from core.model import ExplorerState
from core.attribution import cot_reasoning_trace


def create_reasoning_tab(state: ExplorerState):
    with gr.Tab("Reasoning Trace"):
        gr.Markdown(
            """
            ## Watching the Model Think!

            During chain-of-thought generation, the model produces tokens sequentially. 
            At each step, we project the residual stream activation onto the **Clean 
            direction** to get a scalar signal:

            $$\\text{proj}_t = h_t^{(\\ell)} \\cdot \\hat{d}_{\\text{clean}}$$

            **Positive** projections indicate the model's internal state aligns with the 
            clean direction. **Negative** projections suggest corrupt internal state.
            """
        )

        with gr.Row():
            layer_slider = gr.Slider(
                minimum=0, maximum=50, step=1, value=10,
                label="Layer to Probe",
            )
            prompt_idx = gr.Slider(
                minimum=0, maximum=50, step=1, value=0,
                label="Prompt Index",
            )
            use_clean = gr.Checkbox(value=False, label="Use clean prompt")
            max_tokens = gr.Number(value=80, label="Max Tokens", precision=0)

        trace_btn = gr.Button("Generate & Trace", variant="primary")

        trace_plot = gr.Plot(label="Clean Projection over Generation")
        generated_text = gr.Textbox(label="Generated Text", lines=6, interactive=False)
        stats_text = gr.Textbox(label="Trace Statistics", interactive=False, lines=2)

        def on_trace(layer, idx, clean, max_tok):
            if state.diff_in_means is None:
                return None, "No activations computed", ""

            layer = min(int(layer), state.n_layers - 1)
            idx = int(idx)
            prompts = state.clean_prompts if clean else state.corrupt_prompts
            prompt = prompts[idx] if idx < len(prompts) else prompts[0]

            result = cot_reasoning_trace(state, prompt, layer, max_new_tokens=int(max_tok))

            tokens = result["tokens"]
            projs = result["projections"]

            # Color by sign: positive = clean (blue), negative = corrupt (red)
            colors = ["#4dabf7" if p > 0 else "#ff6b6b" for p in projs]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(len(projs))),
                y=projs,
                marker_color=colors,
                hovertext=[f"'{t}': {p:.3f}" for t, p in zip(tokens, projs)],
                hoverinfo="text",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)

            # Rolling average
            if len(projs) > 5:
                window = min(5, len(projs) // 3)
                rolling = np.convolve(projs, np.ones(window) / window, mode="valid")
                offset = window // 2
                fig.add_trace(go.Scatter(
                    x=list(range(offset, offset + len(rolling))),
                    y=rolling,
                    mode="lines",
                    line=dict(color="#ffd43b", width=2),
                    name=f"Rolling avg (w={window})",
                ))

            fig.update_layout(
                title=f"Clean Projection — Layer {layer}",
                xaxis_title="Generation Step",
                yaxis_title="Projection onto clean direction",
                template="plotly_dark",
                height=450,
            )

            # Stats
            mean_proj = projs.mean()
            std_proj = projs.std()
            frac_positive = (projs > 0).mean()
            trend = np.polyfit(range(len(projs)), projs, 1)[0] if len(projs) > 2 else 0

            prompt_type = "clean" if clean else "corrupt"
            stats = (
                f"Prompt type: {prompt_type}\n"
                f"Mean projection: {mean_proj:.3f} | Std: {std_proj:.3f} | "
                f"Fraction positive: {frac_positive:.1%} | Trend slope: {trend:.4f}"
            )

            return fig, result["generated_text"], stats

        trace_btn.click(
            on_trace,
            inputs=[layer_slider, prompt_idx, use_clean, max_tokens],
            outputs=[trace_plot, generated_text, stats_text],
        )
