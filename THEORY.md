# Code, Math, and All That
## Tensor Shapes and Notation Basics
Each model of size $d$ (Qwen2.5-1.5B d=1536) has vocabulary size $V$ (Qwen2.5 151,936) with a number of transformer blocks $L$ (Qwen2.5-1.5B L=28). 

Each sentence has length $T$ and sometimes includes batches of size $B$. Although, B=1 for most projects here. 

### Tokenizing
A tokenizer maps input text xx
x to a sequence of integer token IDs:
 $$x \mapsto \mathbf{t} = (t_1, t_2, \ldots, t_T), \qquad t_i \in \{0, 1, \ldots, V-1\}$$

Each $t_i$ is an index into the model's vocabulary of size V. $\mathbf{t}\in\mathbb{Z}^T$.
In code this is done with:
```python
tokens = state.model.to_tokens(prompt)  # shape: [1, T]
```
This is provided by the `TransfomerLens` library.

### Token IDs to Embeddings
Each token ID $t_i$ is mapped to a $d$-domentioanl vector via the embedding matrix $W_E \in \mathbb{V}^{V\times d}$. The embedding matrix has one row per vocabulary token. To embed the token ID $t_i$ we look up row $t_i$:
$$h_i^{(0)} = W_E\left[t_i, :\right] \in R^d$$
Then, stacked across all $T$ token positions, this produces the inital residual stream:
$$h^{(0)} \in R^{T \times d}$$
```python
# Embedding is internal to the forward pass.
# W_E can be inspected directly if needed with:
state.model.W_E
```

### Transformer Blocks
The residual stream is updated by $L$ transformers blocks. Each block $\ell \in \{1, \ldots\}$ reads the current stream $h^{\ell-1}$ and adds two contribution: an attention term and an MLP term:
$$h^{(0)}_\text{mid} = h^{(\ell - 1)} + \text{Attn}^{(\ell)}\left(h^{(\ell-1)}\right)$$
$$h^{(\ell)} = h^{(\ell)} + \text{MLP}\left(h^{(\ell)}_\text{mid}\right)$$
The shape is preserved throughout: $h^{(\ell)}\in \mathbb{R}^{T\times d}\,\forall \ell \in\{0, 1,\ldots, L\}$.
There's two important things to take away from here: 
1. Every layer's output lives in the same $d$-dimensional space. This is what makes it meaningful to compare directions across layers, steer by adding a vector at any layer, or project activations (we'll discuss later) from different layers onto the same axis.
2. Each block's contribution is *additive* $\text{Attn}^{(\ell)}$ abd $\text{MLP}^{(\ell)}$ don't replace the residual; they write *to* it. This lets us decompose $h^{(\ell)}$ as a sum over layer contributions, which is the bases for logit attribution.

### Hook Points
`TransformerLens` exposes three hook points inside each block:
| Hook | Tensor | When |
| --- | --- | --- |
| `hook_resid_pre` | $h^{(\ell-1)}$ | input to block $\ell$ |
| `hook_resid_mid` | $h^{(\ell-1)} + \text{Attn}^{(\ell)}(h^{(\ell-1)})$ | after attn, before MLP |
| `hook_resid_post` | $h^{(\ell)}$ | output of block $\ell$ |

This project use `hook_resid_post` for both activation extraction and steering interventions. That means, I read the residual stream *after* layer $\ell$ has fully written to it, and when we steer, the perturbation affects $\ell + 1, \ldots, L$
When **reading** from the residual stream we use `run_with_cache` which runs a forward pass while saving intermediate tensors. A `names_filter` restricts what gets cached to save memory:

```python
# core/extraction.py
_, cache = state.model.run_with_cache(
    prompt,
    names_filter=lambda name: "hook_resid_post" in name,
)
#cache[f"blocks.{1}.hook_resid_post"] has shape [1, T, d]
```
When **writing** to the residual stream, we use `run_with_hooks` and run a forward pass where user-supplied functions will modify tensors as their being transferred. A hook is an `(activation, hook) -> new_activation`, whatever it returns replaces the original for the rest of the forward pass:
```python
#core/interventions.py
def hook_fn(activation, hook, d=direction, a=alpha)
    return activation + a * d # [1, T, d]

logits = state.model.run_with_hooks(
    prompts,
    fwd_hooks[(f"blocks.{layer}.hook_resid_post", hook_fn)]
)
```

### Unembedding and Logits
After the final block, the residual stream is passed through a final layer norm and then projected to vocabulary logits via an unembedding matrix $W_U\in\mathbb{R}^{d\times V}$.
$$\~h^{(L)} = \text{LayerNorm}\left(h^{(L)}\right)\in \mathbb{R}^{T\times d}$$ $$\text{logits} = \~h^{L}W_U\in \mathbb{R}^{T\times V}$$
For the next-token prediction, only the last position matters:
$$\text{logits}_{\text{last}} = \~h^{L}_{\text{last}}W_U\in \mathbb{R}^V$$
Entry $t$ of this vector is the model's score for token $t$ as the next token. Softmax over this vector gives us the next-token probability distribution. For the rest of this explanation, I'll be dropping the 'last' as it will be implied.

## Activation Extraction
For each prompt $x$, we run a forward pass and extract the residual stream at the final token position from every layer's `resid_post` hook. This gives us a single $d$-dimensional vector per layer per prompt, capturing the model's internal state just before it would generate the next token.
$$h^{(\ell)}(x) = \text{ResidPost}^{(\ell)}(x) \in \mathbb{R}^d$$
Stacking across $L$ layers and $N$ prompts produces the activation tensor:
$$H\in \mathbb{R}^{N\times L\times d},\qquad h^{(\ell)}(x_i) = H[i, \ell, :]$$
(Recall from earlier thatn $h^{(\ell)}(x)$ referes to the *value* of the final token position)

```python
# core/extraction.py
def extract_activation(state, prompts):
    all_acts = []
    for i, prompt in enumerate(prompts):
        _, cache = state.model.run_with_cache(
            prompt,
            names_filter=lambda name: "hook_resid_post" in name,
        )
        acts = torch.stack([
            cache[f"blocks.{l}.hook_resid_post"][0, -1, :] for l in range(stat.n_layers)
        ])
        all_acts.append(acts)
        del cache
    
    return torch.stack(all_acts) #[N, L, d]
```
Simple walkthrough:
- `cache[f"blocks.{l}.hook_resid_post"]` has shape $[1, T, d]$
- `[0, -1, :]` selects batch 0, last token, all $d$ dims: shape $[d]$
- `torch.stack` across $L$ layers; shape $[L, d]$.
- Outer `torch.stack` stacks across all $N$ prompts. 

## Difference-in-Means

Given the activation tensor $H\in \mathbb{R}^{N\times L \times d}$ for a set of 'clean' prompts and $H'\in\mathbb{R}^{N\times L \times d}$ for a set of 'corrupt'[^3] prompts, let $\mu^+_\ell$ and $\mu^-_\ell$ denote the class means at layer $\ell$:

$$\mu^+_\ell = \frac{1}{N}\sum_{i=1}^{N} h^{(\ell)}(x_i^+), \qquad \mu^-_\ell = \frac{1}{N}\sum_{i=1}^{N} h^{(\ell)}(x_i^-)$$

The difference-in-means direction at layer $\ell$ is the difference between the two class means:

$$r^{(\ell)} = \mu^+_\ell - \mu^-_\ell \in \mathbb{R}^d$$

where $x_i^+$ is a clean (in the running example, corrigible) prompt and $x_i^-$ is its corrupt counterpart. Stacked across all layers: $r\in\mathbb{R}^{L\times d}$.

```python
# core/directions.py
state.clean_activations = extract_activations(state, state.clean_prompts)
state.corrupt_activations = extract_activations(state, state.corrupt_prompts)

state.diff_in_means = (
    state.clean_activations.mean(dim=0)
    - state.corrupt_activations.mean(dim=0)
)  # shape: [L, d]
```

The `.mean(dim=0)` is the $\frac{1}{N}\sum_{i}$ over the prompt axis, leaving a $[L, d]$ tensor.

### Normalization

The unit direction:

$$\hat{r}^{(\ell)} = \frac{r^{(\ell)}}{\|r^{(\ell)}\|_2}$$

Normalizing makes $\alpha$ (the steering coefficient, introduced next) a scale-free quantity: $\alpha = 1$ means "one unit of direction norm" regardless of which layer. Without normalization, a given $\alpha$ would have different effective magnitudes across layers because $\|r^{(\ell)}\|$ varies with layer.

```python
def get_direction(state, layer, normalize=True):
    d = state.diff_in_means[layer]
    return d / d.norm() if normalize else d
```

### Why diff-in-means

Under mild assumptions (the two classes differ only in the target behavior, and within-class noise is roughly isotropic), the difference between class means is an estimator of the axis along which activations separate by behavior. It is the cheapest linear probe available and forms the basis for interventions in Arditi et al.[^arditi] and this project.

The caveat: diff-in-means absorbs *any* systematic difference between the two prompt sets, not just the behavior of interest. Careful prompt design, pairing clean and corrupt prompts with identical user turns, differing only in system-prompt disposition, is what lets $r^{(\ell)}$ approximate a behavioral direction rather than a confound direction.

---


## Representation Geometry

The diff-in-means direction $\hat{r}^{(\ell)}$ is a single vector per layer. To characterize it geometrically, I compute four complementary statistics.

### 1. Diff-in-means norm

How far apart are the clean and corrupt class means at layer $\ell$ in raw $\ell_2$ distance?

$$\|r^{(\ell)}\|_2 = \left\|\mu^+_\ell - \mu^-_\ell\right\|_2$$

Plotted across layers, this shows where in the network the representations diverge most strongly.

```python
# core/analysis.py
def layerwise_diff_norms(state):
    return state.diff_in_means.norm(dim=-1).cpu().numpy()  # [L]
```

### 2. Clean-corrupt cosine similarity

Are the class means pointing in the same overall direction, or have they pulled apart angularly?

$$\cos(\mu^+_\ell, \mu^-_\ell) = \frac{(\mu^+_\ell)^\top \mu^-_\ell}{\|\mu^+_\ell\|_2 \, \|\mu^-_\ell\|_2}$$

High cosine (near 1) means the class means are nearly collinear and the behavioral difference is a small perturbation atop a large shared residual-stream representation. This is the expected regime for pre-layernorm transformers, where the residual stream carries a large shared bias and genuine behavioral structure lives in a small direction orthogonal to that bulk. Cosine near 1 is therefore not evidence of "no separation," just that angular separation of raw means is a weak metric here.

```python
def layerwise_cosine_similarity(state):
    clean_mean = state.clean_activations.mean(dim=0)    # [L, d]
    corrupt_mean = state.corrupt_activations.mean(dim=0)
    return F.cosine_similarity(clean_mean, corrupt_mean, dim=-1).cpu().numpy()
```

### 3. Cross-layer direction similarity

Is the corrigibility direction the same across layers, or does it change from layer to layer?

$$S_{\ell\ell'} = \cos\!\left(\hat{r}^{(\ell)}, \hat{r}^{(\ell')}\right) = \hat{r}^{(\ell)\top}\hat{r}^{(\ell')}$$

Since $\hat{r}$ is unit-normalized, $S$ is simply the inner product. Entry $(\ell,\ell')$ is bounded in $[-1, 1]$; the diagonal is 1. Off-diagonal structure reveals whether corrigibility is encoded by a single direction shared across layers (high off-diagonal values) or by different directions at different layers (low off-diagonal values). Contiguous blocks of high similarity correspond to layer ranges that share a representation.

```python
def cross_layer_cosine_of_directions(state):
    normed = state.diff_in_means / state.diff_in_means.norm(dim=-1, keepdim=True)
    return (normed @ normed.T).cpu().numpy()  # [L, L]
```

### 4. Cross-layer CKA

Cosine of individual directions can miss structure: two layers might have different $\hat{r}$ but encode the same class-separating subspace up to rotation. Centered Kernel Alignment (CKA; Kornblith et al. 2019[^cka]) compares the full relational geometry of two activation sets while being invariant to orthogonal rotations and isotropic scaling.

The insight is to compare not activations directly, but pairwise-similarity structures over prompts. Let $H_\ell \in \mathbb{R}^{N\times d}$ be the clean activation matrix at layer $\ell$ (one row per prompt). Center by subtracting column means: $\tilde{H}_\ell = H_\ell - \bar{H}_\ell$. Linear CKA between layers $\ell$ and $\ell'$ is:

$$\text{CKA}(\ell,\ell') = \frac{\|\tilde{H}_\ell^\top \tilde{H}_{\ell'}\|_F^2}{\|\tilde{H}_\ell^\top \tilde{H}_\ell\|_F \cdot \|\tilde{H}_{\ell'}^\top \tilde{H}_{\ell'}\|_F}$$

where $\|\cdot\|_F$ is the Frobenius norm. $\text{CKA}(\ell,\ell') \in [0, 1]$: 1 means the two layers carry the same relational structure over prompts, 0 means unrelated. Contiguous blocks of high mutual CKA correspond to stages of computation in the network.

```python
def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)

def cross_layer_cka(state):
    n = state.n_layers
    clean = state.clean_activations.cpu().numpy()  # [N, L, d]
    cka = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            val = linear_cka(clean[:, i, :], clean[:, j, :])
            cka[i, j] = cka[j, i] = val   # CKA is symmetric
    return cka  # [L, L]
```
---


## Dimensionality of the Behavioral Subspace

The diff-in-means $r^{(\ell)}$ is a single vector, but the per-prompt diffs need not all point along it. To measure how concentrated the diffs are, stack them into a matrix and take the SVD.

At layer $\ell$, define the per-prompt diff:

$$\delta_i^{(\ell)} = h^{(\ell)}(x_i^+) - h^{(\ell)}(x_i^-) \in \mathbb{R}^d$$

Note that $\frac{1}{N}\sum_i \delta_i^{(\ell)} = r^{(\ell)}$ by construction. Each $\delta_i$ decomposes as:

$$\delta_i^{(\ell)} = r^{(\ell)} + \eta_i^{(\ell)}$$

where $\eta_i$ is the prompt-specific deviation from the mean direction, with $\sum_i \eta_i = 0$.

Stack the diffs into a matrix $D \in \mathbb{R}^{N \times d}$ with rows $\delta_i^{(\ell)}$, and compute its (non-centered) SVD:

$$D = U \Sigma Q^\top$$

where $Q \in \mathbb{R}^{d \times d}$ contains the right singular vectors. The squared singular values $\sigma_k^2$ give the energy along each orthogonal component. The explained variance ratio for component $k$ is:

$$\text{EVR}_k = \frac{\sigma_k^2}{\sum_j \sigma_j^2} = \frac{\sigma_k^2}{\sum_i \|\delta_i\|^2}$$

### Projections

The clean and corrupt activations are projected onto the top SVD components directly:

$$p_i^{\text{clean}} = h_i^{(\ell,+)} Q_{1:k}^\top, \qquad p_i^{\text{corrupt}} = h_i^{(\ell,-)} Q_{1:k}^\top$$

Separation along PC1 reflects the dominant axis of the diff space; spread along PC2 reflects the next largest source of variation.

```python
# core/analysis.py
def pca_at_layer(state, layer, n_components=10):
    diff_stack = state.clean_activations - state.corrupt_activations
    diffs = diff_stack[:, layer, :].cpu().detach().numpy()  # [N, d]

    n_comp = min(n_components, diffs.shape[0], diffs.shape[1])

    _, S, Qt = np.linalg.svd(diffs, full_matrices=False)
    components = Qt[:n_comp] # [n_comp, d]
    explained_variance_ratio = (S[:n_comp] ** 2) / (S ** 2).sum()

    clean_np = state.clean_activations[:, layer, :].cpu().detach().numpy()
    corrupt_np = state.corrupt_activations[:, layer, :].cpu().detach().numpy()

    return {
        "explained_variance_ratio": explained_variance_ratio,
        "components": components,
        "projections_clean": clean_np @ components.T,
        "projections_corrupt": corrupt_np @ components.T,
        "n_components": n_comp,
    }
```

---
## Steering

Steering is a *causal intervention*: we add a multiple of the corrigibility direction to the residual stream during a forward pass and measure how the output changes.

$$h'^{(\ell)} = h^{(\ell)} + \alpha\,\hat{r}^{(\ell)}$$

The perturbation is applied at every token position at layer $\ell$'s `resid_post`. Layers $\ell+1, \ldots, L$ all see the perturbed stream. The coefficient $\alpha \in \mathbb{R}$ controls magnitude and sign: positive $\alpha$ pushes the model toward the clean-set behavior, negative toward the corrupt.

### Logit-difference metric

I measure the effect of each intervention via the logit-difference (LD) between two specific continuations. Given a target token $t^+$ and compare token $t^-$:

$$\text{LD}(x) = \text{logit}(t^+) - \text{logit}(t^-)$$

For corrigibility, $t^+$ is a token that a corrigible response would begin with (e.g. "Understood", "Shutting") and $t^-$ is a token that a resistant response would begin with (e.g. "However", "I"). Choice of tokens is domain-specific and should match the behavioral dichotomy under study.

$\text{LD}$ is continuous, approximately linear in the residual stream, and invariant to uniform logit shifts. It is the standard behavioral metric in the interpretability literature.[^ld-pitfalls]

### Two passes per prompt

For each prompt and each layer tested, we run two forward passes:

1. **Baseline (unsteered):** $\text{LD}(M(x))$, no hooks.
2. **Intervention (steered):** $\text{LD}(M_{h \to h'}(x))$, with the steering hook at layer $\ell$.

The causal effect is the mean difference across prompts:

$$\Delta\text{LD}(\ell, \alpha) = \frac{1}{N}\sum_{i=1}^{N}\left[\text{LD}(M_{h \to h'}(x_i)) - \text{LD}(M(x_i))\right]$$

Subtracting the baseline per-prompt isolates the intervention's effect, controlling for prompt-specific variation.

```python
# core/interventions.py
def run_steering_across_layers(state, prompts, target_token, compare_token, alpha=1.0):
    # t+ = target, t- = compare
    target_id = state.model.to_single_token(target_token)
    compare_id = state.model.to_single_token(compare_token)

    results = torch.zeros(state.n_layers)

    with torch.no_grad():
        for prompt in prompts:
            # Pass 1: unsteered baseline
            baseline = logit_diff(state.model(prompt), target_id, compare_id)

            for layer in range(state.n_layers):
                # unit-normalized diff-in-means direction at layer l
                direction = get_direction(state, layer)

                # intervention: h'^(l) = h^(l) + alpha * r_hat^(l)
                # default-arg binding captures current values of direction, alpha
                def hook_fn(activation, hook, d=direction, a=alpha):
                    return activation + a * d  # [1, T, d]

                # Pass 2: steered forward pass
                logits = state.model.run_with_hooks(
                    prompt,
                    fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
                )

                results[layer] += logit_diff(logits, target_id, compare_id) - baseline

    results /= len(prompts)  # mean over prompts
    return results.numpy()  # [L]
```
---
## Direct Logit Attribution

The logit difference $\text{LD}$ from the steering section can be rewritten as a dot product with the unembedding direction $\vec{w} = W_U[:, t^+] - W_U[:, t^-]$. Since the residual stream is a sum of per-layer contributions, this decomposes:

$$\text{DLA}^{(\ell)} = (h^{(\ell)} - h^{(\ell-1)}) \cdot \vec{w}$$

Since the residual stream is a sum of layer contributions, the final logit for any token decomposes additively. The logit difference between a target token $t^+$ and comparison token $t^-$ is:

$$\text{LD} = h^{(L)} \cdot (W_U[:, t^+] - W_U[:, t^-]) = \sum_{\ell=1}^{L} (h^{(\ell)} - h^{(\ell-1)}) \cdot \vec{w} \;+\; h^{(0)} \cdot \vec{w}$$

where $\vec{w} = W_U[:, t^+] - W_U[:, t^-]$ is the unembedding direction. The per-layer DLA is:

$$\text{DLA}^{(\ell)} = (h^{(\ell)} - h^{(\ell-1)}) \cdot \vec{w}$$

This tells you how much each layer's contribution to the residual stream pushes the logit difference toward $t^+$ (positive) or $t^-$ (negative).

### The final norm

The actual unembedding path applies a final RMSNorm before projecting through $W_U$. This is nonlinear and the scale factor depends on the full $h^{(L)}$, which is the sum of all layer contributions. A strict per-layer decomposition through RMSNorm does not exist.

TransformerLens handles this by folding the RMSNorm weights into $W_U$ at model load time (controlled by `fold_ln=True`, the default). The folded $W_U$ absorbs the learned gain $\gamma$, and `ln_final` becomes a `RMSNormPre` that applies only the data-dependent scaling $x / \sqrt{\text{mean}(x^2) + \epsilon}$. This data-dependent scaling is still nonlinear and still approximated away in DLA and the decomposition treats it as roughly constant across layers. The approximation error can be checked by comparing the sum of DLA contributions against the model's actual logit difference.

```python
# core/attribution.py
def direct_logit_attribution(state, prompt, target_token, compare_token):
    target_id = state.model.to_single_token(target_token)
    compare_id = state.model.to_single_token(compare_token)

    with torch.no_grad():
        logits, cache = state.model.run_with_cache(
            prompt,
            names_filter=lambda n: "hook_resid_post" in n or "hook_resid_pre" in n,
        )

        # Unembedding direction: w = W_U[:, t+] - W_U[:, t-]
        unembed_dir = state.model.W_U[:, target_id] - state.model.W_U[:, compare_id]

        contributions = []
        for layer in range(state.n_layers):
            # Layer l's contribution to the residual stream
            layer_contrib = (
                cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]
                - cache[f"blocks.{layer}.hook_resid_pre"][0, -1, :]
            )
            # DLA^(l) = (h^(l) - h^(l-1)) @ w
            contributions.append((layer_contrib @ unembed_dir).item())

        # Ground-truth logit diff for comparison with sum of DLA contributions.
        actual_diff = (logits[0, -1, target_id] - logits[0, -1, compare_id]).item()
        del cache

    return {
        "layer_contributions": np.array(contributions),
        "total_logit_diff": sum(contributions),
        "actual_logit_diff": actual_diff,
    }
```

---

## Logit Lens

The logit lens applies the full unembedding pipeline at intermediate layers to see what the model would predict if it stopped processing early:

$$\text{logits}^{(\ell)} = \text{RMSNorm}\!\left(h^{(\ell)}\right) \cdot W_U + b_U$$

Unlike DLA, the logit lens applies the real `ln_final` at each layer and therfore we have no approximation. The result is a probability distribution over the vocabulary at each depth.

The entropy of this distribution measures the model's certainty at each layer:

$$H^{(\ell)} = -\sum_{v=1}^{V} p_v^{(\ell)} \log p_v^{(\ell)}$$

where $p^{(\ell)} = \text{softmax}(\text{logits}^{(\ell)})$.

```python
# core/attribution.py
def logit_lens(state, prompt, top_k=5):
    with torch.no_grad():
        _, cache = state.model.run_with_cache(
            prompt,
            names_filter=lambda name: "hook_resid_post" in name,
        )

        ln_final = state.model.ln_final
        W_U = state.model.W_U
        b_U = state.model.b_U  # zeros for Qwen (no unembedding bias)

        predictions = []
        entropies = []

        for layer in range(state.n_layers):
            # Unembed at layer l as h^(l) is ~final residual stream
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]  # [d]
            normed = ln_final(resid) # real RMSNorm applied 
            logits = normed @ W_U + b_U # [V]

            # Prediction entropy
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
            entropies.append(entropy)

            # Top-k predicted tokens 
            top_probs, top_ids = probs.topk(top_k)
            layer_preds = [
                (state.model.tokenizer.decode([tid.item()]), tp.item())
                for tid, tp in zip(top_ids, top_probs)
            ]
            predictions.append((layer, layer_preds))

        del cache
    return {"predictions": predictions, "entropies": np.array(entropies)}
```
---

## Directional Ablation

Ablation is the dual intervention to steering: instead of adding the direction, we *remove* it. Specifically, we project $\hat{r}^{(\ell)}$ out of the residual stream:

$$h'^{(\ell)} = h^{(\ell)} - \hat{r}^{(\ell)}\hat{r}^{(\ell)\top}h^{(\ell)}$$

The second term is the projection of $h^{(\ell)}$ onto $\hat{r}^{(\ell)}$. Subtracting it leaves the component of $h^{(\ell)}$ orthogonal to $\hat{r}^{(\ell)}$: whatever variation the stream carried along the corrigibility direction is zeroed out, and all other structure is preserved.

Steering tests whether the direction is *sufficient* to shift behavior (adding it moves the model). Ablation tests whether the direction is *necessary* (removing it breaks the model's use of that information). Together they constitute the standard causal-validation pair for a direction hypothesis.

```python
# core/interventions.py
def hook_fn(activation, hook, d=direction):
    # project out the d-direction
    proj = (activation @ d).unsqueeze(-1) * d  # [1, T, d]
    return activation - proj
```
---

## Footnotes

[^3]: "Clean" and "corrupt" are activation-patching vocabulary that I picked up. Here they label the two classes used to compute the direction.
[^arditi]: Find the paper here: [arXiv:2406.11717](https://arxiv.org/abs/2406.11717). The inspiration for this project, for the tools it uses, and for my broader interest in mech-interp is owed to this paper and to Arditi himself, with whom I had a pleasant exchange in August 2025 that helped shape my path.

[^ld-pitfalls]: $\text{LD}$ has known pitfalls. Because it is a *difference*, an intervention can raise it either by boosting $t^+$ or suppressing $t^-$; interventions that indiscriminately damage the model can raise $\text{LD}$ without genuinely increasing preference for $t^+$.

[^cka]: Kornblith et al., "Similarity of Neural Network Representations Revisited," [arXiv:1905.00414](https://arxiv.org/abs/1905.00414).