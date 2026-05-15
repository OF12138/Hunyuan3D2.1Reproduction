"""ToMe-SD primitives: bipartite soft matching with merge + unmerge closures.

Adapted for diffusion DiTs where sequence length must be preserved at block
boundaries. The merge closure reduces tokens; the unmerge closure restores
them so the residual stream stays the original length.

Within a single attention call: merge(x) -> attn -> unmerge(attn_out).
Because attention and per-token MLPs are permutation-invariant, the internal
order of tokens between merge and unmerge does not matter — unmerge restores
each token (or its merged representative) to its original position.
"""
import math
from typing import Callable, Tuple

import torch


def do_nothing(x: torch.Tensor, mode: str = None) -> torch.Tensor:
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    protect_mask: torch.Tensor = None,
) -> Tuple[Callable, Callable]:
    """Returns (merge, unmerge) closures.

    metric: [B, N, C] similarity feature per token (use Xpre or K).
    r:      number of tokens to remove this layer.
    class_token: if True, token 0 is protected (never chosen as a source).
    protect_mask: optional [B, N] bool tensor. True = token is protected from
                  being a merge source (e.g. surface/edge tokens). Only
                  even-indexed tokens can be sources; odd-indexed entries in
                  the mask are ignored.
    """
    protected = 1 if class_token else 0
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf

        if protect_mask is not None:
            src_protect = protect_mask[..., ::2]  # [B, len_a]
            scores = scores.masked_fill(src_protect.unsqueeze(-1), -math.inf)

        node_max, node_idx = scores.max(dim=-1)

        # Clamp r to available (unmasked) sources so we never select
        # protected or class tokens as merge sources.
        n_available = (node_max > -math.inf).sum(dim=-1).min().item()
        r = min(r, int(n_available))
        if r <= 0:
            return do_nothing, do_nothing

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted-average merge. Returns (merged_x, merged_size)."""
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size
