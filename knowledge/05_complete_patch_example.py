"""
Complete self-contained example of applying ToMe to a generic ViT implementation.

Assumes your ViT has:
  - model.blocks: list of transformer blocks
  - block.norm1, block.attn, block.norm2, block.mlp: standard ViT structure
  - block.attn: attention module that returns a single tensor (the attended output)
  - model.cls_token: optional CLS token

After calling apply_patch(model), set model.r to control token reduction.
"""

import math
from typing import Tuple, Callable, List, Union

import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# Import ToMe primitives from this repo
# ─────────────────────────────────────────────
from tome.merge import bipartite_soft_matching, merge_wavg, merge_source
from tome.utils import parse_r


# ─────────────────────────────────────────────
# Step 1: Patched Attention
# Returns (output, metric) instead of just output
# ─────────────────────────────────────────────
class ToMeAttention(nn.Module):
    """
    Drop-in replacement for a standard multi-head self-attention module.
    Adds:
      - proportional attention (optional, controlled by _tome_info["prop_attn"])
      - returns key vectors as similarity metric
    """

    def forward(self, x: torch.Tensor, size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        # Your existing QKV projection code here:
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Proportional attention: bias by log(token_size)
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Metric: mean of key vectors across heads → [B, N, head_dim]
        metric = k.mean(dim=1)

        return x, metric


# ─────────────────────────────────────────────
# Step 2: Patched Block
# Inserts merging between attention and MLP
# ─────────────────────────────────────────────
class ToMeBlock(nn.Module):
    """
    Drop-in replacement for a standard ViT block.
    Inserts bipartite_soft_matching + merge_wavg between attention and MLP.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention (with optional proportional attention)
        attn_size = self._tome_info["size"] if self._tome_info.get("prop_attn", True) else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + x_attn

        # Token merging
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

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# Step 3: Patched Top-Level Transformer
# Resets per-forward-pass state
# ─────────────────────────────────────────────
def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        def forward(self, *args, **kwargs):
            self._tome_info["r"]      = parse_r(len(self.blocks), self.r)
            self._tome_info["size"]   = None
            self._tome_info["source"] = None
            return super().forward(*args, **kwargs)
    return ToMeVisionTransformer


# ─────────────────────────────────────────────
# Step 4: apply_patch — entry point
# ─────────────────────────────────────────────
def apply_patch(
    model: nn.Module,
    trace_source: bool = False,
    prop_attn: bool = True,
    OriginalBlock=None,       # pass the block class to replace
    OriginalAttention=None,   # pass the attention class to replace
):
    """
    Patch a ViT model with ToMe.

    Args:
        model:             The ViT model to patch (modified in-place).
        trace_source:      If True, track which original patches each merged token covers.
                           Enables make_visualization() but adds memory overhead.
        prop_attn:         If True, bias attention by log(token_size).
                           Use True for off-the-shelf eval, False for MAE or trained models.
        OriginalBlock:     The Block class used in this model (auto-detected if None).
        OriginalAttention: The Attention class used in this model (auto-detected if None).

    After patching, set model.r to control tokens removed per layer:
        model.r = 8        # remove 8 tokens per layer (constant schedule)
        model.r = (8, -1)  # decreasing schedule
        model.r = [8]*12   # explicit per-layer list
    """
    if hasattr(model, '_tome_info'):
        return  # already patched

    # Auto-detect block/attention classes if not provided
    if OriginalBlock is None or OriginalAttention is None:
        for module in model.modules():
            name = module.__class__.__name__
            if OriginalBlock is None and 'Block' in name:
                OriginalBlock = module.__class__
            if OriginalAttention is None and 'Attention' in name:
                OriginalAttention = module.__class__

    # Patch top-level model class
    ToMeVisionTransformer = make_tome_class(model.__class__)
    model.__class__ = ToMeVisionTransformer
    model.r = 0

    # Initialize shared state dict
    model._tome_info = {
        "r":             model.r,
        "size":          None,
        "source":        None,
        "trace_source":  trace_source,
        "prop_attn":     prop_attn,
        "class_token":   hasattr(model, 'cls_token') and model.cls_token is not None,
        "distill_token": hasattr(model, 'dist_token') and model.dist_token is not None,
    }

    # Patch blocks and attention modules
    for module in model.modules():
        if isinstance(module, OriginalBlock):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info  # shared reference
        elif isinstance(module, OriginalAttention):
            module.__class__ = ToMeAttention
            module._tome_info = model._tome_info


# ─────────────────────────────────────────────
# Usage example
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import timm
    import tome  # or use the classes above directly

    model = timm.create_model("vit_base_patch16_224", pretrained=True)

    # Option A: use the official tome patch (for timm models)
    tome.patch.timm(model, prop_attn=True)

    # Option B: use the generic patch above (for custom models)
    # apply_patch(model, prop_attn=True)

    model.r = 16  # remove 16 tokens per layer → ~2× speedup on ViT-B

    # Visualize merged tokens
    # apply_patch(model, trace_source=True, prop_attn=True)
    # model.r = 16
    # out = model(img_tensor)
    # vis = tome.make_visualization(pil_img, model._tome_info["source"])

    print("Model patched. model.r =", model.r)
    print("_tome_info keys:", list(model._tome_info.keys()))
