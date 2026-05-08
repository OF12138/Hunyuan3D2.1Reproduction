# ToMe: How to Patch a New ViT Implementation

This guide explains exactly how to apply ToMe to a ViT codebase that isn't already supported. The pattern is the same every time.

---

## Overview

ToMe works by **runtime class replacement** (monkey-patching). You do not modify source files. Instead, `apply_patch` swaps the `__class__` of the model and its submodules to custom subclasses that inject the merging logic.

Three classes need replacing:
1. **The top-level transformer** — to initialize per-forward-pass state
2. **The transformer block** — to call merge after attention
3. **The attention module** — to return the similarity metric alongside the output

---

## Step 1: Define `_tome_info`

All shared state lives in a single dict attached to the model. Every block references the same dict object (not copies).

```python
model._tome_info = {
    "r":            model.r,         # will be overwritten each forward pass
    "size":         None,            # accumulated token sizes, reset each forward
    "source":       None,            # source adjacency matrix, reset each forward
    "trace_source": False,           # True only when you need visualization
    "prop_attn":    True,            # True for off-the-shelf eval, False for MAE/training
    "class_token":  True,            # whether the model has a CLS token
    "distill_token": False,          # True for DeiT with distillation token
    # optional extended config:
    "feature_choice":    "K",        # which QKV feature to use as metric
    "head_aggregation":  "mean",     # how to aggregate over attention heads
}
```

---

## Step 2: Patch the Top-Level Transformer

Subclass the model's top-level class to reset state at the start of every forward pass:

```python
def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        def forward(self, *args, **kwargs):
            self._tome_info["r"]      = parse_r(len(self.blocks), self.r)
            self._tome_info["size"]   = None
            self._tome_info["source"] = None
            return super().forward(*args, **kwargs)
    return ToMeVisionTransformer

ToMeVisionTransformer = make_tome_class(model.__class__)
model.__class__ = ToMeVisionTransformer
model.r = 0  # default: no reduction; user sets this after patching
```

**Critical:** `parse_r` produces a list consumed by `.pop(0)`. If you forget to reinitialize it each forward, the list empties after the first pass and subsequent forwards will error.

---

## Step 3: Patch the Attention Module

Modify attention to:
- Optionally add `log(size)` to attention logits (proportional attention)
- Return the similarity metric (key vectors by default) alongside the output

```python
class ToMeAttention(OriginalAttention):
    def forward(self, x, size=None):
        B, N, C = x.shape
        # ... compute q, k, v as usual ...

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Proportional attention: bias by log of token size
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        # ... dropout, project, etc. ...

        # Return metric: mean of K across heads → [B, N, head_dim]
        metric = k.mean(dim=1)   # dim 1 = heads
        return x_out, metric
```

The metric is what `bipartite_soft_matching` uses to decide which tokens are similar. Key vectors (`k`) work best (see ablations). Head-mean is default; concat is an alternative (larger metric, slightly different behavior).

---

## Step 4: Patch the Transformer Block

Inject the merge between attention and MLP:

```python
class ToMeBlock(OriginalBlock):
    def forward(self, x):
        # --- attention ---
        attn_size = self._tome_info["size"] if self._tome_info.get("prop_attn", True) else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + x_attn   # residual

        # --- token merging ---
        r = self._tome_info["r"].pop(0)
        if r > 0:
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
                tome_info=self._tome_info,
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(merge, x, self._tome_info["source"])
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        # --- MLP ---
        x = x + self.mlp(self.norm2(x))
        return x
```

Key points:
- `metric` comes from the **current** layer's attention, reflecting the post-attention token representations.
- `merge_source` must be called **before** `merge_wavg` (source uses pre-merge `x` for shape info).
- `self._tome_info["size"]` carries over between layers and accumulates — do not reset it here.

---

## Step 5: Wire Everything Together in `apply_patch`

```python
def apply_patch(model, trace_source=False, prop_attn=True):
    # Guard: don't double-patch
    if hasattr(model, '_tome_info'):
        return

    # 1. Patch top-level class
    model.__class__ = make_tome_class(model.__class__)
    model.r = 0

    # 2. Initialize shared state dict
    model._tome_info = {
        "r": model.r, "size": None, "source": None,
        "trace_source": trace_source, "prop_attn": prop_attn,
        "class_token": <check model>, "distill_token": <check model>,
    }

    # 3. Patch every block and attention module
    for module in model.modules():
        if isinstance(module, OriginalBlock):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info   # shared reference, not copy
        elif isinstance(module, OriginalAttention):
            module.__class__ = ToMeAttention
            module._tome_info = model._tome_info
```

After `apply_patch(model)`, the user controls reduction with `model.r = <int>`.

---

## Special Case: MAE Global Average Pooling

When using global average pooling (instead of CLS token), the pool must be size-weighted after ToMe merges tokens:

```python
T = x.shape[1]  # original patch count, captured BEFORE blocks run
# ... run blocks ...
if model._tome_info["size"] is not None:
    x = (x * model._tome_info["size"])[:, 1:, :].sum(dim=1) / T
else:
    x = x[:, 1:, :].mean(dim=1)
```

Capture `T` before the blocks, not after — by then tokens have been merged and `x.shape[1]` is smaller.

---

## Special Case: Non-standard Token Layout

SWAG models store tokens as `(tokens, batch, channels)` instead of `(batch, tokens, channels)`. The `swag.py` patch adds an `Encoder`-level class that transposes around the block calls. If your target architecture uses a different layout, add a similar wrapper.

---

## Checklist for Porting

- [ ] Identify the Block class (one per transformer layer)
- [ ] Identify the Attention class inside the block
- [ ] Identify the top-level model class (has a list of blocks)
- [ ] Check for CLS token and/or distillation token
- [ ] Check if the model uses global average pooling (needs size-weighted pool fix)
- [ ] Check token dimension layout `(B, N, C)` vs other orderings
- [ ] Verify that the attention module gives access to `q`, `k`, `v` separately
- [ ] Set `prop_attn=True` for off-the-shelf eval, `False` for training or MAE
