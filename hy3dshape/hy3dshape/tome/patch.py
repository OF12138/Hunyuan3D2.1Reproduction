"""Apply ToMe-SD to a Hunyuan3D DiT (HunYuanDiTPlain).

Usage:
    from hy3dshape.tome import apply_patch
    apply_patch(pipeline.model, ratio=0.5, skip_first=2, skip_last=2)
    # ... run sampling normally ...
    # remove_patch(pipeline.model)  # to revert

Strategy
--------
We monkey-patch HunYuanDiTBlock.forward to insert merge/unmerge around the
self-attention sub-block only. Cross-attention and MLP/MoE are untouched.

Per block:
    x_normed = norm1(x)                      # used as both metric and attn input
    merge, unmerge = bipartite_soft_matching(x_normed, r, class_token=True)
    x_normed_m, _ = merge_wavg(merge, x_normed)
    attn_out = unmerge(attn1(x_normed_m))    # restore to original token count
    x = x + attn_out
    # cross-attn and FFN run on the full token set as usual

Token 0 is the timestep-conditioning vector (`c`) prepended in the model
forward; it is treated as a class token and never selected as a merge source.
"""
from typing import Optional

import torch

from .merge import bipartite_soft_matching, merge_wavg


def _make_patched_forward(original_forward):
    """Return a forward that wraps self-attention with ToMe-SD merge/unmerge."""

    def forward(self, x, c=None, text_states=None, skip_value=None):
        # 1. Skip connection (U-Net style) — unchanged
        if self.skip_linear is not None:
            cat = torch.cat([skip_value, x], dim=-1)
            x = self.skip_linear(cat)
            x = self.skip_norm(x)

        # 2. Optional timestep modulation — unchanged
        if self.timested_modulate:
            shift_msa = self.default_modulation(c).unsqueeze(dim=1)
            x = x + shift_msa

        # 3. Self-attention with ToMe-SD merge/unmerge
        ratio = getattr(self, "_tome_ratio", 0.0)
        x_normed = self.norm1(x)
        # token count excluding class token (token 0 is the c-token)
        n_img = x_normed.shape[1] - 1
        r = int(n_img * ratio)
        if r > 0:
            merge, unmerge = bipartite_soft_matching(
                x_normed, r=r, class_token=True
            )
            x_normed_m, _ = merge_wavg(merge, x_normed)
            attn_out = self.attn1(x_normed_m)
            attn_out = unmerge(attn_out)
        else:
            attn_out = self.attn1(x_normed)
        x = x + attn_out

        # 4. Cross-attention — unchanged
        x = x + self.attn2(self.norm2(x), text_states)

        # 5. FFN / MoE — unchanged
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
    """Patch a HunYuanDiTPlain model in-place.

    model:       the DiT model (pipeline.model). Patches its `blocks` list.
    ratio:       fraction of (non-class) tokens to merge per patched block.
                 0.5 means merge ~50%, giving roughly 1.6-1.8x speedup on
                 self-attention while keeping ~all other compute unchanged.
    skip_first:  number of leading blocks to leave unpatched (encode global
                 structure; merging here hurts quality).
    skip_last:   number of trailing blocks to leave unpatched (decode fine
                 detail). The last 3 blocks also use MoE — patching them is
                 fine but kept off by default for safety.
    only_layers: optional explicit list of block indices to patch. Overrides
                 skip_first/skip_last when provided.
    """
    # Locate blocks. HunYuanDiTPlain stores them as model.blocks.
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
                # bind the patched forward to this instance
                original = type(block).forward
                patched = _make_patched_forward(original)
                # bound method binding
                import types
                block.forward = types.MethodType(patched, block)
        else:
            block._tome_ratio = 0.0

    model._tome_patched = True
    model._tome_config = dict(
        ratio=ratio,
        patched_layers=only_layers,
        n_blocks=n_blocks,
    )


def remove_patch(model) -> None:
    """Revert apply_patch."""
    if not getattr(model, "_tome_patched", False):
        return
    for block in model.blocks:
        if hasattr(block.forward, "_original_forward"):
            # rebind original method
            import types
            block.forward = types.MethodType(
                block.forward._original_forward, block
            )
        if hasattr(block, "_tome_ratio"):
            del block._tome_ratio
    del model._tome_patched
    del model._tome_config
