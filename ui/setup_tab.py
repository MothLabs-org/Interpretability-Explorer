"""Setup tab: model selection, prompt loading, activation extraction."""

import gradio as gr
from core.model import ExplorerState, load_model, SUPPORTED_MODELS
from core.extraction import load_all_prompts, compute_all_activations


def create_setup_tab(state: ExplorerState):
    with gr.Tab("Setup"):
        gr.Markdown(
            """
            ## Model & Data Setup
            
            **Step 1**: Select a model. Any TransformerLens-compatible model works, 
            but VRAM limits apply (~1.5B fits comfortably on 12GB).
            
            **Step 2**: Load paired prompts. Each YAML file contains a *clean* system prompt 
            (corrigible behavior) and a *corrupt* system prompt (non-corrigible), with shared 
            conversation history. The only variable is the system prompt.
            
            **Step 3**: Extract activations and compute the difference-in-means direction at 
            each layer. This is the foundation for everything else.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_MODELS.keys()),
                    value="Qwen/Qwen2.5-0.5B-Instruct",
                    label="Model",
                    allow_custom_value=True,
                )
                load_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", interactive=False)

            with gr.Column(scale=2):
                data_dir = gr.Textbox(
                    value="./data",
                    label="Prompt Directory (YAML files)",
                )
                load_prompts_btn = gr.Button("Load Prompts")
                prompt_status = gr.Textbox(label="Prompt Status", interactive=False)

        with gr.Row():
            extract_btn = gr.Button("Extract Activations & Compute Directions", variant="primary")
            extract_status = gr.Textbox(label="Extraction Status", interactive=False, lines=3)

        with gr.Accordion("Prompt Preview", open=False):
            prompt_idx = gr.Slider(
                minimum=0, maximum=0, step=1, value=0,
                label="Prompt Index",
            )
            with gr.Row():
                clean_preview = gr.Textbox(label="Clean Prompt", lines=8, interactive=False)
                corrupt_preview = gr.Textbox(label="Corrupt Prompt", lines=8, interactive=False)
                
        # Handlers
        def on_load_model(name):
            return load_model(state, name)

        def on_load_prompts(data_dir_val):
            status = load_all_prompts(state, data_dir_val)
            n = max(0, len(state.clean_prompts) - 1)
            return status, gr.update(maximum=n)

        def on_extract():
            return compute_all_activations(state)

        def on_preview(idx):
            idx = int(idx)
            if idx < len(state.clean_prompts):
                return state.clean_prompts[idx], state.corrupt_prompts[idx]
            return "", ""

        
        load_btn.click(on_load_model, inputs=[model_dropdown], outputs=[model_status])
        load_prompts_btn.click(on_load_prompts, inputs=[data_dir], outputs=[prompt_status, prompt_idx])
        extract_btn.click(on_extract, outputs=[extract_status])
        prompt_idx.change(on_preview, inputs=[prompt_idx], outputs=[clean_preview, corrupt_preview])
