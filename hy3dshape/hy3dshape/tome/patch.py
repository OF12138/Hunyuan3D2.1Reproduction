"""Apply ToMe-SD (+ locality extension) to a Hunyuan3D DiT.

Two strategies are available via apply_patch:

1. Vanilla / surface-aware  (Option 2 from previous iteration)
   -------------------------------------------------------
   Full-block merge/unmerge.  Optionally protects high-activation tokens from
   being merge sources (protect_ratio > 0).

   apply_patch(model, ratio=0.5, protect_ratio=0.3)

2. Locality-aware  (Option 1 – 3D-locality bipartite matching)
   ----------------------------------------------------------
   Uses a trained PositionHead to estimate each token's 3D centroid at
   inference time.  The merge score is a weighted combination of cosine
   feature similarity and a Gaussian spatial proximity term, preventing
   merges across geometrically distant regions of the shape.

   apply_patch(model, ratio=0.5,
               pos_head_path="tome_data/pos_head.pt",
               locality_alpha=0.5,
               locality_sigma=0.3)

Block strategy (both modes)
---------------------------
    # full-token: skip + modulation
    x = skip_norm(skip_linear(cat(skip_value, x)))   [if skip_linear]
    x = x + default_modulation(c)                    [if timested_modulate]

    # merge once
    positions = pos_head(x[:, 1:, :])               [locality mode only]
    merge, unmerge = matching(norm1(x), r, ...)
    x_m = merge_wavg(merge, x)

    # sub-blocks on merged tokens
    x_m = x_m + attn1(norm1(x_m))
    x_m = x_m + attn2(norm2(x_m), text_states)
    x_m = x_m + (moe or mlp)(norm3(x_m))

    return unmerge(x_m)

Token 0 is the timestep-conditioning vector (class token) and is never
selected as a merge source in either mode.
"""
import types
from typing import Optional

import torch

from .merge import bipartite_soft_matching, locality_bipartite_soft_matching, merge_wavg


def _make_patched_forward(original_forward, pos_head=None):
    """Return a patched forward.

    pos_head: if not None, a PositionHead used for locality-aware matching.
    """

    def forward(self, x, c=None, text_states=None, skip_value=None):
        # 1. Skip connection (full-token count to match skip_value shape)
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
        n_img = x.shape[1] - 1          # exclude class token
        r = int(n_img * ratio)

        if r > 0:
            # ---- Optional: locality positions --------------------------------
            if pos_head is not None:
                with torch.no_grad():
                    pos_img = pos_head(x[:, 1:, :].to(
                        next(pos_head.parameters()).dtype
                    )).to(x.dtype)                  # [B, N_img, 3]
                    cls_pos = torch.zeros(
                        x.shape[0], 1, 3,
                        device=x.device, dtype=x.dtype
                    )
                    positions = torch.cat([cls_pos, pos_img], dim=1)  # [B, 1+N_img, 3]

                locality_alpha = getattr(self, "_tome_locality_alpha", 0.5)
                locality_sigma = getattr(self, "_tome_locality_sigma", 0.3)

                metric = self.norm1(x)
                merge, unmerge = locality_bipartite_soft_matching(
                    metric, positions, r=r, class_token=True,
                    alpha=locality_alpha, sigma=locality_sigma,
                )

            # ---- Surface-aware protection (Option 2 fallback) ---------------
            else:
                protect_mask = None
                if protect_ratio > 0:
                    with torch.no_grad():
                        saliency = x[:, 1:, :].norm(dim=-1)    # [B, N_img]
                        protect_k = int(n_img * protect_ratio)
                        if protect_k > 0:
                            _, top_idx = saliency.topk(protect_k, dim=-1)
                            protect_img = torch.zeros_like(saliency, dtype=torch.bool)
                            protect_img.scatter_(-1, top_idx, True)
                            protect_mask = torch.cat([
                                torch.zeros(x.shape[0], 1, dtype=torch.bool,
                                            device=x.device),
                                protect_img,
                            ], dim=1)                          # [B, 1+N_img]

                metric = self.norm1(x)
                merge, unmerge = bipartite_soft_matching(
                    metric, r=r, class_token=True, protect_mask=protect_mask,
                )

            # ---- Sub-blocks on merged tokens ---------------------------------
            x_m, _ = merge_wavg(merge, x)

            x_m = x_m + self.attn1(self.norm1(x_m))
            x_m = x_m + self.attn2(self.norm2(x_m), text_states)

            mlp_inputs = self.norm3(x_m)
            if self.use_moe:
                x_m = x_m + self.moe(mlp_inputs)
            else:
                x_m = x_m + self.mlp(mlp_inputs)

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
    # Option 2: surface-aware protection
    protect_ratio: float = 0.0,
    # Option 1: locality-aware matching
    pos_head_path: Optional[str] = None,
    locality_alpha: float = 0.5,
    locality_sigma: float = 0.3,
) -> None:
    """Patch a HunYuanDiTPlain model in-place.

    model:           DiT model (pipeline.model).  Patches its `blocks` list.
    ratio:           Fraction of image tokens to merge per patched block.
    skip_first:      Leading blocks to leave unpatched (coarse structure blocks).
    skip_last:       Trailing blocks to leave unpatched (fine-detail blocks).
    only_layers:     Explicit list of block indices to patch (overrides skip_*).
    protect_ratio:   (Option 2) Fraction of high-saliency tokens immune to merging.
    pos_head_path:   (Option 1) Path to a trained PositionHead .pt file.
                     If set, enables locality-aware matching (overrides protect_ratio).
    locality_alpha:  (Option 1) Proximity weight in combined score (0=feature, 1=position).
    locality_sigma:  (Option 1) Spatial bandwidth for Gaussian proximity kernel.
                     Shape space is ≈ [-1,1]³, so sigma=0.3 ≈ 15% of the diagonal.
    """
    if not hasattr(model, "blocks"):
        raise AttributeError(
            "apply_patch expects model.blocks (HunYuanDiTPlain). "
            f"Got model of type {type(model).__name__}."
        )

    # Load position head if requested (hidden_size auto-detected from weights)
    pos_head = None
    if pos_head_path is not None:
        from .pos_head import PositionHead
        device = next(model.parameters()).device
        pos_head = PositionHead.load(pos_head_path, device=str(device))
        pos_head = pos_head.to(device)
        print(f"[ToMe-locality] loaded PositionHead from {pos_head_path} "
              f"(hidden_size={pos_head.hidden_size})")

    n_blocks = len(model.blocks)
    if only_layers is None:
        only_layers = list(range(skip_first, n_blocks - skip_last))

    for i, block in enumerate(model.blocks):
        block._tome_layer_idx = i
        if i in only_layers:
            block._tome_ratio = ratio
            block._tome_protect_ratio = protect_ratio
            block._tome_locality_alpha = locality_alpha
            block._tome_locality_sigma = locality_sigma
            if not hasattr(block.forward, "_original_forward"):
                original = type(block).forward
                patched = _make_patched_forward(original, pos_head=pos_head)
                block.forward = types.MethodType(patched, block)
        else:
            block._tome_ratio = 0.0
            block._tome_protect_ratio = 0.0

    if pos_head is not None:
        model._tome_pos_head = pos_head   # keep alive on the model

    mode = "locality" if pos_head is not None else (
        "surface-aware" if protect_ratio > 0 else "vanilla"
    )
    model._tome_patched = True
    model._tome_config = dict(
        ratio=ratio,
        patched_layers=only_layers,
        n_blocks=n_blocks,
        mode=mode,
        protect_ratio=protect_ratio,
        locality_alpha=locality_alpha if pos_head is not None else None,
        locality_sigma=locality_sigma if pos_head is not None else None,
        pos_head_path=pos_head_path,
    )


def remove_patch(model) -> None:
    """Revert apply_patch, restoring original block forwards."""
    if not getattr(model, "_tome_patched", False):
        return
    for block in model.blocks:
        if hasattr(block.forward, "_original_forward"):
            block.forward = types.MethodType(
                block.forward._original_forward, block
            )
        for attr in ("_tome_ratio", "_tome_protect_ratio",
                     "_tome_locality_alpha", "_tome_locality_sigma"):
            if hasattr(block, attr):
                delattr(block, attr)
    for attr in ("_tome_patched", "_tome_config", "_tome_pos_head"):
        if hasattr(model, attr):
            delattr(model, attr)
