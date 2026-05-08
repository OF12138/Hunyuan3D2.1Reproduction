# ToMe: Mathematical Foundations

## Problem Statement

A Vision Transformer (ViT) processes an image as a sequence of `N` patch tokens, each of shape `[B, N, C]`. The self-attention cost is O(N²) per layer. ToMe reduces `N` by `r` tokens per layer using a lossless-by-design merge — merged tokens carry combined information rather than being discarded.

---

## 1. Bipartite Soft Matching

### Partition

Tokens are split into two equally-sized sets **A** and **B**:

- **A** = tokens at even indices:  `metric[:, ::2, :]`   — shape `[B, N/2, C]`
- **B** = tokens at odd indices:   `metric[:, 1::2, :]`  — shape `[B, N/2, C]`

This is called the **alternating** partition (other options: sequential, random — but alternating is the default and best from ablations).

### Similarity Scoring

Each token in A is scored against every token in B using **cosine similarity**:

```
metric = metric / ||metric||    # L2-normalize along channel dim
scores = A @ B^T                # [B, N/2, N/2]
```

Cosine similarity is the default and best-performing distance function (outperforms Euclidean, dot, softmax-normalized).

If a **class token** exists (index 0, always in A), its row is set to `-inf` so it is never chosen as a source to merge.

### Greedy Matching

Select the `r` most-similar pairs:

```
node_max, node_idx = scores.max(dim=-1)    # best B-partner for each A-token
edge_idx = node_max.argsort(descending=True)  # rank A-tokens by their best score

src_idx = edge_idx[:r]     # top-r A-tokens will be merged (sources)
unm_idx = edge_idx[r:]     # remaining A-tokens stay unmerged
dst_idx = node_idx[src_idx]  # which B-token each source merges into
```

Constraint: `r ≤ (N - protected) // 2` — at most 50% of tokens can be removed per layer (since sources come only from A).

### After Merging

The resulting token count is:
```
N_new = (N/2 - r)   # unmerged A-tokens
       + N/2         # all B-tokens (destinations)
      = N - r
```

---

## 2. Weighted Average Merging (merge_wavg)

Naive averaging loses information when a token has been merged multiple times. ToMe tracks each token's **size** (number of original patches it represents) and uses it as a weight.

### Forward merge

```
# On first call, size = ones([B, N, 1])  (each token represents 1 patch)
x_merged   = scatter_reduce(x * size, dst_idx, reduce="sum")
size_merged = scatter_reduce(size,   dst_idx, reduce="sum")
x_merged   = x_merged / size_merged   # weighted average
```

The `size` tensor grows as merges accumulate across layers. By the last layer, a single token may represent dozens of original patches — the size weights ensure the merged representation stays correct.

### Unmerge (for reconstruction/visualization)

The reverse operation scatters merged representations back to the original N positions. Used only in `trace_source` mode for visualization, not in normal inference.

---

## 3. Proportional Attention

After merging, tokens have unequal sizes. Standard attention treats all tokens equally. Proportional attention biases attention logits by log(size) so that larger (merged) tokens receive attention proportional to how many original patches they represent:

```
attn_logits = Q @ K^T / sqrt(d)
if size is not None:
    attn_logits += log(size)[:, None, None, :, 0]  # broadcast over heads
attn = softmax(attn_logits)
```

This is only needed for **off-the-shelf** (non-trained) evaluation. When training with ToMe, the model learns to handle unequal token sizes naturally — so `prop_attn=False` during training and MAE fine-tuning.

---

## 4. r Schedule (parse_r)

`r` controls how many tokens are removed per layer. It can be:

| Type | Meaning |
|------|---------|
| `int` | Same `r` every layer |
| `(r, inflection)` | Linearly varying schedule; inflection ∈ [-1, 1]. `(r, 0)` = constant; `(r, -1)` = decreasing (remove more early, fewer late) |
| `list[int]` | Explicit per-layer values |

The decreasing schedule `(r, -1)` is what the paper calls the "decreasing schedule" and can outperform constant `r` in some settings. Internally, `parse_r` always converts to a `list[int]` of length = num_layers, which is consumed layer by layer via `.pop(0)`.

---

## 5. Source Tracking (for visualization)

When `trace_source=True`, an adjacency matrix `source` of shape `[B, N, N_orig]` is maintained. It records which original patches each merged token covers. Initially it is the identity matrix `I_N`. After each merge layer:

```
source = scatter_reduce(source, dst_idx, reduce="amax")
```

At the end, `source.argmax(dim=1)` gives the group assignment for each original patch — used to color-code the visualization image.

---

## 6. MAE Global Pool Correction

Standard MAE uses global average pooling over all non-CLS tokens: `x[:, 1:].mean(dim=1)`. After ToMe, tokens have unequal sizes, so the pool must be size-weighted:

```
# T = original number of patches (before any merging)
x_pooled = (x * size)[:, 1:, :].sum(dim=1) / T
```

This maintains the correct weighted mean over the original patches and is critical for accuracy when using MAE models with global pooling.
