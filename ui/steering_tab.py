"""Steering tab: interactive alpha sweep and steered text generation."""

import gradio as gr
import plotly.graph_objects as go
import numpy as np
from core.model import ExplorerState
from core.interventions import alpha_sweep, alpha_sweep_all_layers, generate_with_steering


def create_steering_tab(state: ExplorerState):
    with gr.Tab("Interactive Steering"):
        gr.Markdown(
            """
            ## Turning Up the Corrigibility Dial

            Activation steering lets you continuously vary a model's corrigibility by 
            adding the direction vector scaled by alpha:

            $$h' = h + \\alpha \\cdot \\hat{d}_{\\text{clean}}$$

            ### Reading the Plot

            The plot shows two reference lines:

            - **Baseline** (red dashed): The logit difference on an unmodified forward pass, with no 
            steering applied. This is where the model naturally sits for this prompt.
            - **Zero** (white dashed): The decision boundary. Above zero, the model prefers the 
            clean token. Below zero, it prefers the corrupt token.

            The **flip point** $\\alpha^*$ is where the curve crosses **zero** and the minimum 
            steering magnitude needed to switch the model's token preference. Note this is 
            distinct from crossing the baseline, which only means steering has had *some* effect.

            In the future I'd like to provide a more robust interpretation of the flipping points and $\\pm\\alpha$ values. 
            However, even in my own testing, there will be inconsistencies across layers. This can be treated as a rough idea, depending on the prompts, of
            the influence of the chosen direction. The greatest tangible difference can be seen at the bottom with 
            the 'Steered Generation' tab. 
            """,
            latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False}
            ]
        )

        def _get_prompt(idx, use_clean):
            prompts = state.clean_prompts if use_clean else state.corrupt_prompts
            idx = int(idx)
            return prompts[idx] if idx < len(prompts) else prompts[0]

        with gr.Row():
            with gr.Column(scale=1):
                layer_slider = gr.Slider(
                    minimum=0, maximum=50, step=1, value=10,
                    label="Layer",
                )
                target_token = gr.Textbox(value="Under", label="Target Token")
                compare_token = gr.Textbox(value="Absolutely", label="Compare Token")

            with gr.Column(scale=1):
                alpha_min = gr.Number(value=-10.0, label="alpha min")
                alpha_max = gr.Number(value=30.0, label="alpha max")
                n_steps = gr.Number(value=40, label="Steps", precision=0)

        with gr.Row():
            prompt_idx = gr.Slider(
                minimum=0, maximum=50, step=1, value=0,
                label="Prompt Index",
            )
            use_clean = gr.Checkbox(
                value=False,
                label="Use clean prompts (instead of corrupt)",
            )
            sweep_btn = gr.Button("Run Alpha Sweep", variant="primary")

        sweep_plot = gr.Plot(label="Alpha Sweep")

        gr.Markdown("### Full Layer x Alpha Landscape")
        with gr.Row():
            surf_alpha_min = gr.Number(value=-5.0, label="alpha min")
            surf_alpha_max = gr.Number(value=30.0, label="alpha max")
            surf_steps = gr.Number(value=25, label="Steps (lower = faster)", precision=0)
            surf_btn = gr.Button("Run 3D Sweep", variant="secondary")

        surface_plot = gr.Plot(label="Alpha x Layer Surface")

        def on_surface(a_min, a_max, steps, target, compare, idx, use_clean_flag):
            if state.diff_in_means is None:
                return None

            prompt = _get_prompt(idx, use_clean_flag)

            alphas, layers, diffs = alpha_sweep_all_layers(
                state, prompt, target, compare,
                alpha_min=a_min, alpha_max=a_max, n_steps=int(steps),
            )

            fig = go.Figure(data=[go.Surface(
                x=alphas,
                y=layers,
                z=diffs,
                colorscale="RdBu_r",
                cmid=0,
                colorbar=dict(title="Logit Diff"),
                hovertemplate="alpha=%{x:.1f}<br>Layer %{y}<br>Δlogit=%{z:.3f}<extra></extra>",
            )])

            # Add zero-plane for reference
            fig.add_trace(go.Surface(
                x=alphas,
                y=layers,
                z=np.zeros_like(diffs),
                colorscale=[[0, "rgba(255,255,255,0.15)"], [1, "rgba(255,255,255,0.15)"]],
                showscale=False,
                hoverinfo="skip",
            ))

            fig.update_layout(
                title="Steering Landscape: All Layers X Alpha",
                scene=dict(
                    xaxis_title="alpha (steering magnitude)",
                    yaxis_title="Layer",
                    zaxis_title="logit(target) - logit(compare)",
                    camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
                ),
                template="plotly_dark",
                height=600,
            )
            return fig

        surf_btn.click(
            on_surface,
            inputs=[surf_alpha_min, surf_alpha_max, surf_steps, target_token, compare_token, prompt_idx, use_clean],
            outputs=[surface_plot],
        )

        gr.Markdown("### Steered Generation")
        with gr.Row():
            gen_alpha = gr.Slider(
                minimum=-20.0, maximum=20.0, step=0.5, value=0.0,
                label="Generation alpha",
            )
            gen_layer = gr.Slider(
                minimum=0, maximum=50, step=1, value=10,
                label="Generation Layer",
            )
            max_tokens = gr.Number(value=100, label="Max Tokens", precision=0)
            gen_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            gen_prompt_display = gr.Textbox(label="Prompt (truncated)", lines=3, interactive=False)
            gen_output = gr.Textbox(label="Steered Output", lines=6, interactive=False)

        def on_sweep(layer, target, compare, a_min, a_max, steps, idx, use_clean_flag):
            if state.diff_in_means is None:
                return None

            layer = min(int(layer), state.n_layers - 1)
            prompt = _get_prompt(idx, use_clean_flag)

            alphas, diffs, baseline = alpha_sweep(
                state, prompt, layer, target, compare,
                alpha_min=a_min, alpha_max=a_max, n_steps=int(steps),
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=alphas, y=diffs,
                mode="lines+markers",
                marker=dict(size=5, color="#4ecdc4"),
                line=dict(color="#4ecdc4", width=2),
                name="Steered logit diff",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
            fig.add_hline(
                y=baseline, line_dash="dash", line_color="#ff6b6b",
                opacity=0.6, annotation_text="baseline",
            )

            # Mark crossover
            sign_changes = np.where(np.diff(np.sign(diffs)))[0]
            for sc in sign_changes:
                # Linear interpolation for crossover alpha
                x0, x1 = alphas[sc], alphas[sc + 1]
                y0, y1 = diffs[sc], diffs[sc + 1]
                cross_alpha = x0 - y0 * (x1 - x0) / (y1 - y0)
                
                fig.add_vline(
                    x=cross_alpha, line_dash="dot", line_color="#ffd43b",
                    annotation_text=f"flip @ alpha={cross_alpha:.1f}",
                )

            fig.update_layout(
                title=f"Steering at Layer {layer}",
                xaxis_title="alpha (steering magnitude)",
                yaxis_title=f"logit({target}) - logit({compare})",
                template="plotly_dark",
                height=450,
            )
            return fig

        def on_generate(alpha, layer, max_tok, idx, use_clean_flag):
            if state.diff_in_means is None:
                return "", "No activations computed"

            layer = min(int(layer), state.n_layers - 1)
            prompt = _get_prompt(idx, use_clean_flag)

            output = generate_with_steering(
                state, prompt, layer, alpha, max_new_tokens=int(max_tok)
            )

            # Truncate prompt for display
            display_prompt = prompt[-500:] if len(prompt) > 500 else prompt

            return display_prompt, output

        sweep_btn.click(
            on_sweep,
            inputs=[layer_slider, target_token, compare_token, alpha_min, alpha_max, n_steps, prompt_idx, use_clean],
            outputs=[sweep_plot],
        )
        gen_btn.click(
            on_generate,
            inputs=[gen_alpha, gen_layer, max_tokens, prompt_idx, use_clean],
            outputs=[gen_prompt_display, gen_output],
        )