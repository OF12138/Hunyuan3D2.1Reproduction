# ToMe Knowledge Base

Reference materials for applying Token Merging (ToMe) to a new Vision Transformer project.

## Files

| File | What it covers |
|------|---------------|
| `01_math.md` | The mathematics: bipartite matching, weighted average merge, proportional attention, r schedules, MAE global pool fix |
| `02_core_code.md` | Annotated source of `tome/merge.py` — the matching kernel, merge/unmerge closures, `merge_wavg`, `merge_source`, `parse_r` |
| `03_patching_guide.md` | Step-by-step instructions for patching any ViT implementation, including special cases (MAE global pool, non-standard token layout) |
| `04_design_decisions.md` | Ablation findings from Table 1: which feature/distance/combine/partition/prop_attn settings to use and why |
| `05_complete_patch_example.py` | Self-contained working example of `apply_patch` with all three class replacements wired together |

## Quick reference

```python
import tome

# 1. Patch the model (timm ViT example)
tome.patch.timm(model, prop_attn=True)   # off-the-shelf eval
# tome.patch.timm(model, prop_attn=False) # MAE or trained with ToMe

# 2. Set reduction per layer
model.r = 8   # int, (int, float) schedule, or list[int]

# 3. Use normally — no other changes needed
output = model(image_tensor)

# 4. Visualization (requires trace_source=True at patch time)
# vis = tome.make_visualization(pil_image, model._tome_info["source"])
```

## Key invariants to never break

1. `_tome_info["r"]` must be re-initialized as a fresh list at the start of **every** forward pass (done in `ToMeVisionTransformer.forward`). It is consumed via `.pop(0)` each block.
2. All blocks share the **same** `_tome_info` dict object — not copies. The `size` accumulates across blocks.
3. Call `merge_source` **before** `merge_wavg` within a block (source uses pre-merge `x` shape).
4. Use `prop_attn=True` for off-the-shelf, `prop_attn=False` for training or MAE fine-tuned models.
5. For MAE with global average pooling, capture `T = x.shape[1]` before the blocks run, then pool as `(x * size)[:, 1:].sum(1) / T`.
