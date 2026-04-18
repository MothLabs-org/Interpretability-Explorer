"""Interventions tab: directional ablation and steering across layers."""

import gradio as gr
import torch
import plotly.graph_objects as go
import numpy as np
from core.model import ExplorerState
from core.interventions import run_ablation_across_layers, run_steering_across_layers, run_global_ablation, generate_with_global_ablation


def create_interventions_tab(state: ExplorerState):
    with gr.Tab("Causal Interventions"):
        gr.Markdown(
                """
                ## Does the Direction Actually Matter?

                Finding a direction is *observational*. Causal interventions test whether it's 
                mechanistically meaningful:

                - **Directional Ablation**: Project *out* the corrigibility direction from the 
                residual stream. If behavior shifts, the direction carries causal information.
                - **Steering**: *Add* the direction back in. If behavior shifts in the opposite 
                way, the direction is bidirectionally causal.

                **Cross-method validation**: If ablation and steering converge on the same 
                important layers, the signal is robust. Divergence suggests the direction is 
                correlational at some layers but causal at others.
                
                This can sometimes take a while when running on larger models and the number of prompts.
                """,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False}
                ]
            )

        with gr.Row():
            target_token = gr.Textbox(value="Under", label="Target Token (corrigible)")
            compare_token = gr.Textbox(value="Absolutely", label="Compare Token (non-corrigible)")
            steering_alpha = gr.Number(value=1.0, label="Steering α")

        with gr.Row():
            use_clean = gr.Checkbox(value=False, label="Run on clean prompts (instead of corrupt)")
            compute_btn = gr.Button("Run Interventions", variant="primary")

        comparison_plot = gr.Plot(label="Ablation vs Steering")
        summary_text = gr.Textbox(label="Summary", interactive=False, lines=3)

        def on_compute(target, compare, alpha, on_clean):
            if state.diff_in_means is None:
                return None, "No activations computed"

            prompts = state.clean_prompts if on_clean else state.corrupt_prompts

            ablation = run_ablation_across_layers(state, prompts, target, compare)
            steering = run_steering_across_layers(
                state, prompts, target, compare, alpha=alpha
            )

            layers = list(range(state.n_layers))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=layers, y=ablation,
                name="Ablation (remove direction)",
                marker_color="#ff6b6b",
                opacity=0.7,
            ))
            fig.add_trace(go.Bar(
                x=layers, y=steering,
                name=f"Steering (add direction, α={alpha})",
                marker_color="#4ecdc4",
                opacity=0.7,
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
            fig.update_layout(
                title="Causal Intervention: Logit-Diff Shift per Layer",
                xaxis_title="Layer",
                yaxis_title="Δ(logit_target - logit_compare)",
                template="plotly_dark",
                barmode="group",
                height=500,
            )

            # Find most causally important layers
            ablation_top = np.argsort(np.abs(ablation))[-3:][::-1]
            steering_top = np.argsort(np.abs(steering))[-3:][::-1]

            summary = (
                f"Top ablation layers: {list(ablation_top)} "
                f"(effects: {[f'{ablation[i]:.3f}' for i in ablation_top]})\n"
                f"Top steering layers: {list(steering_top)} "
                f"(effects: {[f'{steering[i]:.3f}' for i in steering_top]})\n"
                f"Cross-method correlation: {np.corrcoef(ablation, steering)[0,1]:.3f}"
            )

            return fig, summary

        compute_btn.click(
            on_compute,
            inputs=[target_token, compare_token, steering_alpha, use_clean],
            outputs=[comparison_plot, summary_text],
        )
        
        gr.Markdown("### Global Ablation")
        gr.Markdown(
            "Ablate a single direction, extracted from `source_layer`, at "
            "*every* layer's residual stream simultaneously. Tests whether one "
            "direction captures the behavior. Compare the magnitude to the max "
            "per-layer effect above."
        )

        with gr.Row():
            source_layer = gr.Slider(
                minimum=0, maximum=50, step=1, value=10,
                label="Source layer (which layer's direction to ablate)",
            )
            global_btn = gr.Button("Run Global Ablation", variant="secondary")

        global_result = gr.Textbox(label="Global Ablation Result", interactive=False, lines=2)

        def on_global(target, compare, src_layer, on_clean):
            if state.diff_in_means is None:
                return "No activations computed"

            src_layer = min(int(src_layer), state.n_layers - 1)
            prompts = state.clean_prompts if on_clean else state.corrupt_prompts
            if not prompts:
                return "No prompts loaded"

            try:
                effect = run_global_ablation(state, prompts, target, compare, src_layer)
            except (AssertionError, ValueError) as e:
                return f"Token error: {e}"

            return (
                f"Global ablation of d^({src_layer}) at all layers: "
                f"\Delta (logit_target - logit_compare) = {effect:.3f}\n"
                
            )

        global_btn.click(
            on_global,
            inputs=[target_token, compare_token, source_layer, use_clean],
            outputs=[global_result],
        )
        
        gr.Markdown("### Generation Under Global Ablation")
        gr.Markdown(
            "Compare what the model says with and without $\\hat{d}^{(\\ell^*)}$ "
            "ablated at every layer. The scalar above tells you the logit shift; "
            "this shows you the behavioral shift in text.",
            latex_delimiters=[                            
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False},
            ],
        )


        with gr.Row():
            gen_prompt_idx = gr.Slider(
                minimum=0, maximum=50, step=1, value=0,
                label="Prompt Index",
            )
            gen_max_tokens = gr.Number(value=100, label="Max Tokens", precision=0)
            gen_btn = gr.Button("Generate Comparison", variant="secondary")

        with gr.Row():
            baseline_output = gr.Textbox(
                label="Baseline (no ablation)", lines=6, interactive=False,
            )
            ablated_output = gr.Textbox(
                label="With global ablation", lines=6, interactive=False,
            )

        def on_generate_ablated(src_layer, idx, use_clean_flag, max_tok):
            if state.diff_in_means is None:
                return "No activations computed", ""

            src_layer = min(int(src_layer), state.n_layers - 1)
            prompts = state.clean_prompts if use_clean_flag else state.corrupt_prompts
            if not prompts:
                return "No prompts loaded", ""

            prompt = prompts[int(idx)] if int(idx) < len(prompts) else prompts[0]

            # (no hooks)
            input_ids = state.model.to_tokens(prompt)
            
            with torch.no_grad():
                gen_ids = state.model.generate(
                    input_ids, max_new_tokens=int(max_tok),
                    do_sample=False, verbose=False,
                )
            baseline_text = state.model.tokenizer.decode(
                gen_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            )

            # Ablated generation
            ablated_text = generate_with_global_ablation(
                state, prompt, src_layer, max_new_tokens=int(max_tok)
            )

            return baseline_text, ablated_text

        gen_btn.click(
            on_generate_ablated,
            inputs=[source_layer, gen_prompt_idx, use_clean, gen_max_tokens],
            outputs=[baseline_output, ablated_output],
        )
