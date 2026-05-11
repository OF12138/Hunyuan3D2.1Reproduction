# ToMe-SD on Hunyuan3D Stage 1 — v1

This note describes **where** token merging is injected into the Stage-1
shape DiT, **how** the injection works, and **why** each choice was made.
The implementation lives in `hy3dshape/hy3dshape/tome/` and is enabled at
runtime — no model weights are touched.

---

## 1. Stage 1 in one paragraph

Stage 1 is the image → latent-shape flow-matching DiT in
`hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline`. The denoiser is
`HunYuanDiTPlain` (`hy3dshape/hy3dshape/models/denoisers/hunyuandit.py`),
a U-Net–shaped stack of `HunYuanDiTBlock`s. Per step, the model:

1. Embeds the latent into tokens via `x_embedder`, builds a conditioning
   vector `c` from the timestep + pooled image embedding.
2. **Prepends** `c` as token 0: `x = torch.cat([c, x], dim=1)`
   (line 657 of `hunyuandit.py`). The sequence length is therefore
   `1 + N_latent` (with `N_latent = 4096` at the default resolution),
   i.e. **token 0 is a class-like conditioning token, tokens 1..N are
   spatial latent tokens.**
3. Runs `depth` blocks (`self.blocks`) with U-Net skip connections: the
   first half push activations into `skip_value_list`, the second half
   pop them and fuse via `skip_linear + skip_norm`.

Each `HunYuanDiTBlock.forward(x, c, text_states, skip_value)` does:

```
skip  → skip_linear(cat) → skip_norm        (only if skip_linear is not None)
add   timestep modulation                   (if timested_modulate)
attn1 self-attention,  residual
attn2 cross-attention with text_states,  residual
mlp / moe,  residual
```

All three sub-blocks (self-attn, cross-attn Q-side, MLP/MoE) are
**per-token**, so they scale linearly with sequence length. That is the
opening for token merging.

---

## 2. Where ToMe-SD is injected

### File map

| File | Role |
|---|---|
| `hy3dshape/hy3dshape/tome/merge.py` | `bipartite_soft_matching` + `merge_wavg` primitives (return **merge** and **unmerge** closures). |
| `hy3dshape/hy3dshape/tome/patch.py` | Monkey-patches `HunYuanDiTBlock.forward` with a ToMe-SD wrapper. Public API: `apply_patch(model, ratio, skip_first, skip_last)` / `remove_patch(model)`. |
| `hy3dshape/hy3dshape/tome/__init__.py` | Re-exports `apply_patch`, `remove_patch`. |
| `demo_tome_full.py` | End-to-end demo (Stage 1 + Stage 2) with `--tome / --no-tome / --ratio / --profile / --tag`. |

### Injection point

We do **not** edit `hunyuandit.py`. Instead, `apply_patch` walks
`model.blocks` and replaces the `forward` method of selected blocks
with a ToMe-SD wrapper (monkey-patch via `types.MethodType`). This:

* keeps the original code path untouched and reversible
  (`remove_patch` restores it),
* avoids any change to weights / config,
* lets us pick exactly **which** blocks to patch and **how much** to
  merge from the call site.

By default we skip the first 2 and last 2 blocks
(`skip_first=2, skip_last=2`), patching the middle U-Net trunk. The
endpoints are the most sensitive to token resolution (early blocks
build features, the final block emits SDF latents), and per ToMe-SD
practice those are the ones where merging hurts quality most.

---

## 3. Core design — full-block ToMe-SD

ToMe (image classification) merges tokens *monotonically* — sequence
length shrinks layer by layer. That doesn't work here, because:

* the DiT has **U-Net skip connections**: a deep block's input must
  match the stored `skip_value` token count from the symmetric early
  block,
* downstream of the DiT, the **Shape VAE / surface extractor** expects
  the full `1 + N_latent` token sequence to decode an SDF grid.

So we use the **ToMe-SD pattern**: per block, *merge → run sub-blocks
on the smaller set → unmerge back to the original sequence*. Token
order between merge and unmerge does not matter because attention and
per-token MLP/MoE are permutation-invariant.

We wrap the **whole block** in one merge/unmerge pair (not one pair per
sub-block) because the dominant cost is the MLP/MoE, not self-attention
— self-attention already uses FlashAttention. Merging once and then
running attn1 + attn2 + MLP/MoE on the merged set gives ~`(1 - ratio)`
compute on **all three** sub-blocks for the price of a single
merge/unmerge.

### Patched forward (the core code)

`hy3dshape/hy3dshape/tome/patch.py`:

```python
def forward(self, x, c=None, text_states=None, skip_value=None):
    # 1) Skip + timestep modulation must run on full tokens, because
    #    skip_value was stored at the original sequence length.
    if self.skip_linear is not None:
        cat = torch.cat([skip_value, x], dim=-1)
        x = self.skip_linear(cat)
        x = self.skip_norm(x)
    if self.timested_modulate:
        shift_msa = self.default_modulation(c).unsqueeze(dim=1)
        x = x + shift_msa

    ratio = getattr(self, "_tome_ratio", 0.0)
    n_img = x.shape[1] - 1          # exclude token 0 (class/c token)
    r = int(n_img * ratio)

    if r > 0:
        # 2) Build merge/unmerge once. Use norm1(x) as the similarity
        #    metric ("Xpre") — same feature the next attn would see.
        metric = self.norm1(x)
        merge, unmerge = bipartite_soft_matching(
            metric, r=r, class_token=True
        )
        x_m, _ = merge_wavg(merge, x)             # weighted merge of residual stream

        # 3) All per-token sub-blocks run on the merged tensor.
        x_m = x_m + self.attn1(self.norm1(x_m))
        x_m = x_m + self.attn2(self.norm2(x_m), text_states)
        mlp_inputs = self.norm3(x_m)
        if self.use_moe:
            x_m = x_m + self.moe(mlp_inputs)
        else:
            x_m = x_m + self.mlp(mlp_inputs)

        # 4) Restore original token count for skip storage / next block.
        return unmerge(x_m)

    # ratio = 0 → original path.
    x = x + self.attn1(self.norm1(x))
    x = x + self.attn2(self.norm2(x), text_states)
    mlp_inputs = self.norm3(x)
    if self.use_moe:
        x = x + self.moe(mlp_inputs)
    else:
        x = x + self.mlp(mlp_inputs)
    return x
```

### Bipartite soft matching (with unmerge)

`hy3dshape/hy3dshape/tome/merge.py`. Same algorithm as the original
ToMe paper, but we **also return an `unmerge` closure**:

* split tokens into `a = x[..., ::2, :]` (src side) and
  `b = x[..., 1::2, :]` (dst side),
* compute cosine similarity `a @ b.T` on the chosen metric,
* mask out token 0 in `a` so it cannot be a merge source
  (`class_token=True` → `scores[..., 0, :] = -inf`),
* pick the top-`r` highest-similarity src tokens, scatter-reduce them
  into their best dst,
* concatenate the surviving `unm` and the merged `dst` for `merge`,
* the `unmerge` closure writes `unm`, `dst`, and a `dst`-gathered copy
  back into their original even/odd slots.

`merge_wavg(merge, x)` does a size-weighted merge so merging two tokens
acts like an average over how many original tokens each currently
represents — important when the same block is part of a series.

---

## 4. Why these specific choices

| Decision | Reason |
|---|---|
| Monkey-patch instead of editing `hunyuandit.py` | Cleanly reversible (`remove_patch`), doesn't risk a regression in training/eval code that imports the same class. |
| `class_token=True` to protect token 0 | Token 0 carries the timestep + image conditioning vector. Merging it into a spatial token would smear conditioning across the latent grid. |
| Wrap **whole block**, not just attn1 | Profiling pass-1 (attn1-only ToMe) gave ~4% wall-time win; MLP/MoE is the real bottleneck (MoE has top-k gating across 8 experts × 4× hidden). Wrapping the whole block redirects the saving to where the cost lives. |
| `norm1(x)` as similarity metric | "Xpre" — the activation the next attention would see anyway; widely reported to be a strong, cheap similarity signal in ToMe-SD. |
| Skip first 2 / last 2 blocks | Early and late blocks are the most sensitive to spatial fidelity (input embed, final SDF emission). Merging in the trunk preserves quality best. |
| `merge_wavg` (not plain mean) | Maintains correct effective weight per token across blocks, since the residual stream accumulates contributions of different effective sizes. |
| Merge/unmerge **once per block** | One bipartite-matching pass per block instead of three; cheaper bookkeeping, and lets the unmerge boundary fall on the residual stream so the skip connection (stored from a different patched block) still receives full-length tokens. |

---

## 5. How to use it

```python
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.tome import apply_patch, remove_patch

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2.1"
)
apply_patch(pipeline.model, ratio=0.5, skip_first=2, skip_last=2)
mesh = pipeline(image=img)[0]
# remove_patch(pipeline.model)  # to revert
```

End-to-end comparison and profiling are exposed in `demo_tome_full.py`:

```bash
python demo_tome_full.py --no-tome --tag baseline
python demo_tome_full.py --tome --ratio 0.5 --tag tome50
python demo_tome_full.py --tome --ratio 0.7 --tag tome70
python demo_tome_full.py --no-tome --tag baseline --profile   # writes <tag>_stage1_profile.json
```

Each run prints Stage-1 / Stage-2 wall-time and peak VRAM and writes
`<tag>_shape.glb` (untextured) and `<tag>_textured.glb` (final) so the
geometric *and* textured outputs can be compared between `baseline` and
`tomeXX` runs.

---

## 6. Status

* **v1**: full-block ToMe-SD on the trunk blocks, configurable `ratio`,
  skip-first/skip-last endpoints preserved, U-Net skip connections
  preserved, class token protected.
* Open question for v2: whether the remaining floor is **Volume
  Decoding** in the Shape VAE (~31 s at default settings). If the
  profiler confirms, the next lever is `pipeline.enable_flashvdm()`
  (or equivalent), which is orthogonal to ToMe.
