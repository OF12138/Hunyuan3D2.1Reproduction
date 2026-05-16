"""ToMe-SD primitives: bipartite soft matching with merge + unmerge closures.

Two matching strategies are available:

bipartite_soft_matching
    Classic ToMe: cosine similarity only.  Selects the top-r even→odd merge
    pairs by feature similarity.  Optionally protects high-saliency tokens
    from being merge sources (Option 2, surface-aware).

locality_bipartite_soft_matching
    Option 1 (3D-locality): combines cosine similarity with a spatial
    proximity score derived from predicted 3D token centroids.
    score(a, b) = (1-alpha) * cos(a, b) + alpha * exp(-||pa-pb||² / 2σ²)
    This prevents merging tokens that represent geometrically distant
    surfaces even when they look similar in feature space.

Both functions return (merge, unmerge) closures that preserve the original
token count at block boundaries.
"""
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


def do_nothing(x: torch.Tensor, mode: str = None) -> torch.Tensor:
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    protect_mask: Optional[torch.Tensor] = None,
) -> Tuple[Callable, Callable]:
    """Returns (merge, unmerge) closures.

    metric:       [B, N, C] similarity feature per token (use Xpre or K).
    r:            number of tokens to remove this layer.
    class_token:  if True, token 0 is protected (never chosen as a source).
    protect_mask: optional [B, N] bool – True means token cannot be a src.
                  Used by Option 2 (surface-aware protection).
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

        # Clamp r to the number of actually-selectable source tokens
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


def locality_bipartite_soft_matching(
    metric: torch.Tensor,
    positions: torch.Tensor,
    r: int,
    class_token: bool = False,
    alpha: float = 0.5,
    sigma: float = 0.3,
) -> Tuple[Callable, Callable]:
    """Locality-aware bipartite soft matching (Option 1).

    Combines cosine feature similarity with a spatial proximity score derived
    from predicted 3D token centroids, preventing geometrically distant tokens
    from being merged even when their feature vectors are similar.

    score(a, b) = (1 - alpha) * cos(feat_a, feat_b)
                 + alpha      * exp(-||pos_a - pos_b||² / (2 * sigma²))

    metric:     [B, N, D]  token features for cosine similarity.
    positions:  [B, N, 3]  predicted 3D centroids in normalised shape space.
    r:          number of tokens to remove.
    class_token: if True, token 0 is never a merge source.
    alpha:      weight given to spatial proximity (0 = feature-only, 1 = position-only).
    sigma:      spatial bandwidth; tokens > ~2σ apart get near-zero proximity score.
                With shape space ≈ [-1,1]³, sigma=0.3 means ~15% of the diagonal.
    """
    protected = 1 if class_token else 0
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        # ---- Feature similarity (cosine) ------------------------------------
        feat = F.normalize(metric.float(), dim=-1)
        a_feat = feat[..., ::2, :]    # [B, la, D]
        b_feat = feat[..., 1::2, :]   # [B, lb, D]
        cos_scores = a_feat @ b_feat.transpose(-1, -2)   # [B, la, lb]

        # ---- Spatial proximity ----------------------------------------------
        a_pos = positions[..., ::2, :].float()    # [B, la, 3]
        b_pos = positions[..., 1::2, :].float()   # [B, lb, 3]
        # ||pa - pb||²: [B, la, lb]
        diff = a_pos.unsqueeze(-2) - b_pos.unsqueeze(-3)   # [B, la, lb, 3]
        dist_sq = (diff * diff).sum(dim=-1)                  # [B, la, lb]
        proximity = (-dist_sq / (2.0 * sigma * sigma)).exp()

        # ---- Combined score -------------------------------------------------
        scores = (1.0 - alpha) * cos_scores + alpha * proximity

        if class_token:
            scores[..., 0, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        n_available = (node_max > -math.inf).sum(dim=-1).min().item()
        r = min(r, int(n_available))
        if r <= 0:
            return do_nothing, do_nothing

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    # Capture N (original token count) for unmerge
    N = metric.shape[1]

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
        out = torch.zeros(n, N, c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    return merge, unmerge
