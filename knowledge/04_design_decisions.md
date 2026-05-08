# ToMe: Design Decisions and Ablation Findings

This file records what the paper's ablation experiments (Table 1) established about each design choice, so you know which defaults to use and when to deviate.

---

## Summary of Default Configuration

```python
feature_choice   = "K"           # use key vectors as similarity metric
distance_func    = "cosine"      # cosine similarity for token matching
head_aggregation = "mean"        # average K across attention heads
combine_method   = "weighted avg" # weighted average when merging token values
partition_style  = "alternating"  # even/odd token split
prop_attn        = True          # for off-the-shelf; False for trained/MAE models
```

---

## Table 1(a): Feature Choice

**Which representation to use as the similarity metric for matching.**

| Feature | Description |
|---------|-------------|
| `K` | Key vectors from attention (default) |
| `Q` | Query vectors |
| `V` | Value vectors |
| `Xpre` | Token representation before attention (`norm1(x)`) |
| `X` | Token representation after attention + residual |

**Finding:** `K` is the best-performing feature. It reflects what each token "attends to" and is already computed for free during the attention pass — no extra cost.

**Implementation note:** In `ToMeAttention.forward`, the chosen feature is returned as the second return value. `ToMeBlock` passes it directly to `bipartite_soft_matching` as `metric`.

---

## Table 1(b): Distance Function

**How to measure similarity between token pairs.**

| Function | Formula |
|----------|---------|
| `cosine` | `(A @ B^T)` after L2-normalization — default |
| `dot` | `A @ B^T` without normalization |
| `eucl` | Negative Euclidean distance |
| `softmax` | Softmax-normalized dot product |

**Finding:** Cosine similarity is best. It normalizes for feature magnitude, which matters because token norms vary. The L2 normalization is applied inside `bipartite_soft_matching` and has negligible compute cost.

---

## Table 1(c): Head Aggregation

**How to combine the metric (e.g., K) across multiple attention heads.**

| Method | Description |
|--------|-------------|
| `mean` | Average K across heads → `[B, N, head_dim]` — default |
| `concat` | Concatenate K from all heads → `[B, N, C]` |

**Finding:** Mean and concat perform similarly. Mean is preferred because it keeps the metric dimension small, reducing the cost of computing `scores = A @ B^T`.

---

## Table 1(d): Combining Method

**What to do with token values when two tokens are merged.**

| Method | Description |
|--------|-------------|
| `weighted avg` | Weighted average by accumulated size — default |
| `avg pool` | Simple average (ignores size) |
| `max pool` | Take element-wise max |
| `keep one` | Keep only the destination token, discard the source |

**Finding:** Weighted average is significantly better than the others. It correctly accounts for how many original patches each token represents and is what `merge_wavg` implements.

---

## Table 1(e): Partition Style

**How to split N tokens into two sets A and B for bipartite matching.**

| Style | Description |
|-------|-------------|
| `alternating` | A = even indices, B = odd indices — default |
| `sequential` | A = first half, B = second half |
| `random` | Randomly assigned each forward pass |

**Finding:** Alternating is best and most stable. It interleaves spatially adjacent tokens across the two sets, ensuring local patches are evenly represented. Random performs comparably but adds variance. Sequential is worst.

**Note:** The `random` variant is implemented separately as `random_bipartite_soft_matching` in `merge.py`. The default `bipartite_soft_matching` always uses alternating.

---

## Table 1(f): Proportional Attention

**Whether to bias attention logits by `log(token_size)`.**

| Setting | When to use |
|---------|-------------|
| `prop_attn=True` | Off-the-shelf evaluation of pre-trained models |
| `prop_attn=False` | Training with ToMe, or evaluating MAE-finetuned models |

**Finding:** Proportional attention helps significantly for off-the-shelf models that were never exposed to merged tokens during training. These models have no way to "know" that some tokens represent more original patches than others. Adding `log(size)` as an additive bias to attention logits is a zero-training correction that mimics what a trained model would learn.

For models trained with ToMe (or fine-tuned with it), `prop_attn=False` is correct — the model has already learned to handle size-imbalanced tokens.

---

## Choosing `r`: Accuracy-Speed Tradeoff

`r` is the most important hyperparameter. Typical values from the paper:

| Model | r | Speed gain | Accuracy drop |
|-------|---|-----------|---------------|
| ViT-S/16 | 13 | ~1.6× | ~2.1% |
| ViT-B/16 | 13 | ~1.65× | ~2.0% |
| ViT-L/16 | 7–8 | ~1.75× | ~1.5% |
| ViT-H/14 | 7 | ~1.8× | ~0.4% |

Larger models tolerate higher `r` better because they have more redundancy per layer. Start at `r=8` for ViT-L and tune from there.

The decreasing schedule `(r, -1)` can outperform constant `r` at the same average reduction — early layers remove more, later layers remove less.

---

## What ToMe Is and Is Not

**ToMe merges**: retains information from all tokens (in a weighted combination). No information is permanently discarded within a layer.

**Token pruning discards**: permanently removes tokens, losing their information.

This distinction is why ToMe can safely merge foreground tokens (high semantic content) in addition to background tokens, while pruning methods must be conservative about what they remove.

**No training required** for the core mechanism, but training with ToMe further recovers accuracy. The off-the-shelf drop is mainly from attention distribution mismatch (partially corrected by `prop_attn`).
