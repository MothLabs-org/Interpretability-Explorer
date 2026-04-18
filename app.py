"""
opens at: http://localhost:5000
"""
import gradio as gr
from core.model import ExplorerState
from ui.setup_tab import create_setup_tab
from ui.token_selection_tab import create_token_selection_tab
from ui.geometry_tab import create_geometry_tab
from ui.dimensionality_tab import create_dimensionality_tab
from ui.interventions_tab import create_interventions_tab
from ui.steering_tab import create_steering_tab
from ui.reasoning_tab import create_reasoning_tab
from ui.attribution_tab import create_attribution_tab
from ui.notes_tab import create_notes_tab

state = ExplorerState()

HEADER = """
# Interpretability Explorer
An interactive tool for mechanistic interpretability of language models.
This tool investigates whether alignment-relevant behaviors
correspond to measurable changes in a model's internal representations. Each tab provides 
a different lens for analyzing the same underlying phenomenon.

Start in the **Setup** tab to load a model and extract activations. The default setting is based on my own *weak* corrigibility runs and prompts.

**NOTE**: If you try to run one of the tabs and nothing appear, ensure you click "Extract Activations and Computer Directions".
"""

def build_app():
    with gr.Blocks(
        title="Interpretability Explorer",
        theme=gr.themes.Monochrome(
            neutral_hue="stone",      
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#FAF7F2",     
            block_background_fill="#F2EDE6",     
            block_border_color="#E0D9CF",       
            body_text_color="#2C2825",          
            block_label_text_color="#6B6560",   
        ),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .prose { max-width: none !important; }
        """,
    ) as app:
        gr.Markdown(HEADER)
        create_setup_tab(state)
        create_geometry_tab(state)
        create_dimensionality_tab(state)
        create_token_selection_tab(state)
        create_attribution_tab(state)        
        create_steering_tab(state)
        create_interventions_tab(state)
        create_reasoning_tab(state)
        create_notes_tab()
    return app

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )