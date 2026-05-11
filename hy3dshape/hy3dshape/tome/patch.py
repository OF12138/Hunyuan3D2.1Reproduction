"""Apply ToMe-SD to a Hunyuan3D DiT (HunYuanDiTPlain).

Usage:
    from hy3dshape.tome import apply_patch
    apply_patch(pipeline.model, ratio=0.5, skip_first=2, skip_last=2)
    # ... run sampling normally ...
    # remove_patch(pipeline.model)  # to revert

Strategy (full-block ToMe-SD)
-----------------------------
We monkey-patch HunYuanDiTBlock.forward to wrap the **entire block** in a
single merge/unmerge pair. All per-token compute (self-attention,
cross-attention query side, MLP/MoE) runs on the merged token set; only
the skip and modulation operate in full-token space because the stored
skip_value has the original token count.

Per block:
    # full-token: skip + modulation
    if skip_linear: x = skip_norm(skip_linear(cat(skip_value, x)))
    if timested_modulate: x = x + default_modulation(c)

    # merge once
    metric = norm1(x)                                  # Xpre as similarity feature
    merge, unmerge = bipartite_soft_matching(metric, r, class_token=True)
    x_m, _ = merge_wavg(merge, x)                      # merge residual stream

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
        n_img = x.shape[1] - 1   # exclude class token
        r = int(n_img * ratio)

        if r > 0:
            # 3. Single merge for the whole block
            metric = self.norm1(x)
            merge, unmerge = bipartite_soft_matching(
                metric, r=r, class_token=True
            )
            x_m, _ = merge_wavg(merge, x)

            # 4. All sub-blocks on merged tokens
            x_m = x_m + self.attn1(self.norm1(x_m))
            x_m = x_m + self.attn2(self.norm2(x_m), text_states)

            mlp_inputs = self.norm3(x_m)
            if self.use_moe:
                x_m = x_m + self.moe(mlp_inputs)
            else:
                x_m = x_m + self.mlp(mlp_inputs)

            # 5. Unmerge back to original token count for skip / next block
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
    only_layers: Optional[list] = None,
) -> None:
    """Patch a HunYuanDiTPlain model in-place (full-block ToMe-SD).

    model:       the DiT model (pipeline.model). Patches its `blocks` list.
    ratio:       fraction of (non-class) tokens to merge per patched block.
                 With full-block wrap, ratio=0.5 means ~50% of compute saved
                 across attn1, attn2 (Q-side), and MLP/MoE.
    skip_first:  number of leading blocks to leave unpatched.
    skip_last:   number of trailing blocks to leave unpatched.
    only_layers: optional explicit list of block indices to patch.
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
            if not hasattr(block.forward, "_original_forward"):
                original = type(block).forward
                patched = _make_patched_forward(original)
                block.forward = types.MethodType(patched, block)
        else:
            block._tome_ratio = 0.0

    model._tome_patched = True
    model._tome_config = dict(
        ratio=ratio,
        patched_layers=only_layers,
        n_blocks=n_blocks,
        mode="full-block",
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
        if hasattr(block, "_tome_ratio"):
            del block._tome_ratio
    del model._tome_patched
    del model._tome_config
