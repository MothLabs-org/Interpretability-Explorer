"""Token Selection tab: find best (t+, t-) tokens per prompt pair."""
import gradio as gr
import pandas as pd
from core.model import ExplorerState
from core.token_selection import (
    aggregate_scores_across_pairs,
    top_k_candidates,
)


def create_token_selection_tab(state: ExplorerState):
    with gr.Tab("Token Selection"):
        gr.Markdown(
        """
        ## Which tokens should we use for the logit-difference metric?

        The LD metric compares $\\text{logit}(t^+) - \\text{logit}(t^-)$. Picking
        these tokens by hand can introduce experimenter bias and doesn't scale
        across projects. This tab surfaces tokens the model itself treats as
        *discriminative* between clean and corrupt prompts, aggregated across all pairs.

        There are limits to this. A small score gap across all prompts does *not*
        mean only two tokens are associated with the direction and it may mean many
        tokens are nearly tied. One fix is to sweep across the top-$k$ candidates
        when scores cluster; this program doesn't yet support that. If the top
        scores are close, treat the choice of token as provisional and check across a range.

        **Scoring methods:**
        - `prob`: raw probab
        ility difference $p^+(t) - p^-(t)$.
        - `logprob`: log-probability difference $\\log p^+(t) - \\log p^-(t)$.
        Same ranking as the raw logit difference for a fixed prompt pair.
        - `kl`: per-token contribution to $D_{KL}(p^+ \\| p^-)$.
        Weighted by how often the token is actually sampled under $p^+$.

        **Interpretation:** Positive scores favor $t^+$; negative scores favor $t^-$.
        """,
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$", "right": "$", "display": False},
            ],
        )

        with gr.Row():
            method = gr.Radio(
                choices=["logprob", "prob", "kl"],
                value="prob",
                label="Scoring method",
            )
            top_k = gr.Slider(
                minimum=5, maximum=30, value=10, step=1,
                label="Top-k to display",
            )

        compute_btn = gr.Button("Compute Token Scores", variant="primary")

        with gr.Row():
            t_plus_table = gr.Dataframe(
                headers=["token_id", "token_str", "score"],
                label="Top candidates for t+",
                wrap=True,
            )
            t_minus_table = gr.Dataframe(
                headers=["token_id", "token_str", "score"],
                label="Top candidates for t-",
                wrap=True,
            )

        status = gr.Markdown("")

        def on_compute(method_choice, k):
            if not state.clean_prompts or not state.corrupt_prompts:
                return None, None, "_Load prompts in the Setup tab first._"
            
            if state.model is None:
                return None, None, "_Load a model in the Setup tab first._"

            scores = aggregate_scores_across_pairs(
                state,
                state.clean_prompts,
                state.corrupt_prompts,
                method=method_choice,
            )
            t_plus, t_minus = top_k_candidates(scores, state, k=int(k))

            df_plus = pd.DataFrame(t_plus)
            df_minus = pd.DataFrame(t_minus)

            msg = (
                f"Computed scores over {len(state.clean_prompts)} prompt pairs "
                f"using `{method_choice}`. Use the top token string (with its leading "
                f"space, if any) in the steering/ablation tabs."
            )
            return df_plus, df_minus, msg

        compute_btn.click(
            on_compute,
            inputs=[method, top_k],
            outputs=[t_plus_table, t_minus_table, status],
        )