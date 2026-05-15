"""Apply ToMe-SD to a Hunyuan3D DiT (HunYuanDiTPlain).

Usage:
    from hy3dshape.tome import apply_patch
    apply_patch(pipeline.model, ratio=0.5, protect_ratio=0.3)
    # ... run sampling normally ...
    # remove_patch(pipeline.model)  # to revert

Strategy (full-block ToMe-SD with surface-aware protection)
-----------------------------------------------------------
We monkey-patch HunYuanDiTBlock.forward to wrap the **entire block** in a
single merge/unmerge pair. All per-token compute (self-attention,
cross-attention query side, MLP/MoE) runs on the merged token set; only
the skip and modulation operate in full-token space because the stored
skip_value has the original token count.

Surface-aware protection (Option 2): The ShapeVAE was trained with
importance sampling that concentrates tokens on edges/corners. High-
activation tokens carry more geometric information. Before merging, the
top ``protect_ratio`` tokens by L2-norm are masked so they can never be
selected as merge *sources*. The merge budget ``r`` is applied only to
the unprotected pool. This prevents the merge from collapsing adjacent
corner/edge tokens that look similar in feature space but are individually
critical for the final SDF.

Per block:
    # full-token: skip + modulation
    if skip_linear: x = skip_norm(skip_linear(cat(skip_value, x)))
    if timested_modulate: x = x + default_modulation(c)

    # saliency-based protection
    saliency = ‖x[:, 1:, :]‖₂                         # per-token L2 norm
    protect_mask = top-k by saliency                   # True = protected

    # merge once (protected tokens excluded from src pool)
    metric = norm1(x)
    merge, unmerge = bipartite_soft_matching(metric, r, class_token=True,
                                             protect_mask=protect_mask)
    x_m, _ = merge_wavg(merge, x)

    # all sub-blocks on merged tokens
    x_m = x_m + attn1(norm1(x_m))
    x_m = x_m + attn2(norm2(x_m), text_states)
    x_m = x_m + (moe or mlp)(norm3(x_m))

    # unmerge once -> 4097 tokens for skip storage / next block input
    return unmerge(x_m)

Token 0 is the timestep-conditioning vector (`c`) prepended in the model
forward; it is treated as a class token and never selected as a merge source.
"""
import types
from typing import Optional

import torch

from .merge import bipartite_soft_matching, merge_wavg


def _make_patched_forward(original_forward):
    """Return a forward that wraps the whole block with ToMe-SD merge/unmerge."""

    def forward(self, x, c=None, text_states=None, skip_value=None):
        # 1. Skip connection (must run on original token count to match skip_value)
        if self.skip_linear is not None:
            cat = torch.cat([skip_value, x], dim=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        # 2. Timestep modulation (full-token)
        if self.timested_modulate:
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa

        ratio = getattr(self, "_tome_ratio", 0.0)
        protect_ratio = getattr(self, "_tome_protect_ratio", 0.0)
        n_img = x.shape[1] - 1   # exclude class token
        r = int(n_img * ratio)

        if r > 0:
            # 3. Surface-aware protection mask
            protect_mask = None
            if protect_ratio > 0:
                with torch.no_grad():
                    saliency = x[:, 1:, :].norm(dim=-1)       # [B, N_img]
                    protect_k = int(n_img * protect_ratio)
                    if protect_k > 0:
                        _, top_idx = saliency.topk(protect_k, dim=-1)
                        protect_img = torch.zeros_like(saliency, dtype=torch.bool)
                        protect_img.scatter_(-1, top_idx, True)
                        protect_mask = torch.cat([
                            torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device),
                            protect_img,
                        ], dim=1)                              # [B, 1+N_img]

            # 4. Single merge for the whole block
            metric = self.norm1(x)
            merge, unmerge = bipartite_soft_matching(
                metric, r=r, class_token=True, protect_mask=protect_mask,
            )
            x_m, _ = merge_wavg(merge, x)

            # 5. All sub-blocks on merged tokens
            x_m = x_m + self.attn1(self.norm1(x_m))
            x_m = x_m + self.attn2(self.norm2(x_m), text_states)

            mlp_inputs = self.norm3(x_m)
            if self.use_moe:
                x_m = x_m + self.moe(mlp_inputs)
            else:
                x_m = x_m + self.mlp(mlp_inputs)

            # 6. Unmerge back to original token count for skip / next block
            return unmerge(x_m)

        # No merging: original path
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), text_states)
        mlp_inputs = self.norm3(x)
        if self.use_moe:
            x = x + self.moe(mlp_inputs)
        else:
            x = x + self.mlp(mlp_inputs)
        return x

    forward._original_forward = original_forward
    return forward


def apply_patch(
    model,
    ratio: float = 0.5,
    skip_first: int = 2,
    skip_last: int = 2,
    protect_ratio: float = 0.0,
    only_layers: Optional[list] = None,
) -> None:
    """Patch a HunYuanDiTPlain model in-place (full-block ToMe-SD).

    model:          the DiT model (pipeline.model). Patches its `blocks` list.
    ratio:          target fraction of image tokens to merge per patched block.
                    Automatically clamped to the available source pool (after
                    protection), so ratio=0.7 + protect=0.3 won't over-merge.
    skip_first:     number of leading blocks to leave unpatched.
    skip_last:      number of trailing blocks to leave unpatched.
    protect_ratio:  fraction of tokens shielded from merging by activation
                    saliency (L2 norm). 0.0 = vanilla ToMe (no protection).
                    0.3 = top 30% tokens by norm are never merge sources.
    only_layers:    optional explicit list of block indices to patch.
    """
    if not hasattr(model, "blocks"):
        raise AttributeError(
            "apply_patch expects model.blocks (HunYuanDiTPlain). "
            f"Got model of type {type(model).__name__}."
        )

    n_blocks = len(model.blocks)
    if only_layers is None:
        only_layers = list(range(skip_first, n_blocks - skip_last))

    for i, block in enumerate(model.blocks):
        if i in only_layers:
            block._tome_ratio = ratio
            block._tome_protect_ratio = protect_ratio
            if not hasattr(block.forward, "_original_forward"):
                original = type(block).forward
                patched = _make_patched_forward(original)
                block.forward = types.MethodType(patched, block)
        else:
            block._tome_ratio = 0.0
            block._tome_protect_ratio = 0.0

    model._tome_patched = True
    model._tome_config = dict(
        ratio=ratio,
        protect_ratio=protect_ratio,
        patched_layers=only_layers,
        n_blocks=n_blocks,
        mode="full-block-surface-aware" if protect_ratio > 0 else "full-block",
    )


def remove_patch(model) -> None:
    """Revert apply_patch."""
    if not getattr(model, "_tome_patched", False):
        return
    for block in model.blocks:
        if hasattr(block.forward, "_original_forward"):
            block.forward = types.MethodType(
                block.forward._original_forward, block
            )
        for attr in ("_tome_ratio", "_tome_protect_ratio"):
            if hasattr(block, attr):
                delattr(block, attr)
    del model._tome_patched
    del model._tome_config
