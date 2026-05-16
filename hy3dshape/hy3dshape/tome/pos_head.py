"""PositionHead: linear ℝᵈ → ℝ³ that predicts each DiT latent token's 3D centroid.

Training supervision (free):
    The VAE geo_decoder cross-attends grid query points (each with known xyz)
    against the latent tokens. Token i's centroid is:

        c_i = Σ_q  softmax(Qq · Ki / √d)_q  ×  xyz_q

    A Linear(d, 3) is then trained to regress c_i from DiT block hidden states.

Inference (negligible cost):
    head(x)  →  positions [B, N, 3]
    Run once per block, then pass positions to locality_bipartite_soft_matching.

Important: features must come from the **DiT** (block hidden state, 2048-dim in
Hunyuan3D-2.1), NOT the VAE (1024-dim).  Although both are token spaces over
the same latent grid, the dimensionalities are independent.  We capture DiT
hidden states during a real inference run and pair them with VAE-derived
centroids of the final clean z_0.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionHead(nn.Module):
    """Single linear layer ℝ^d → ℝ^3 predicting normalised 3D token centroid."""

    def __init__(self, hidden_size: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, 3)
        # Small init so early predictions are near-origin rather than random
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] → positions: [B, N, 3] in shape space [-1, 1]^3."""
        return self.proj(x)

    @classmethod
    def load(cls, path: str, hidden_size: Optional[int] = None,
             device: str = 'cpu') -> 'PositionHead':
        """Load a saved head.  If hidden_size is None, auto-detect from weights."""
        state = torch.load(path, map_location=device, weights_only=True)
        if hidden_size is None:
            # weight shape is [out=3, in=hidden_size]
            hidden_size = int(state["proj.weight"].shape[1])
        head = cls(hidden_size=hidden_size)
        head.load_state_dict(state)
        return head.eval()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)


# ---------------------------------------------------------------------------
# Centroid extraction – used only during offline training data generation
# ---------------------------------------------------------------------------

class _CentroidCaptureProcessor:
    """Replaces geo_decoder's attn_processor to capture raw attention weights.

    Unlike F.scaled_dot_product_attention (which uses a fused kernel that
    doesn't expose weights), this explicitly forms softmax(QKᵀ/√d) and
    accumulates the centroid sums on-the-fly to avoid storing the full
    [B, H, N_q, N_kv] matrix.
    """

    def __init__(self, xyz: torch.Tensor):
        """xyz: [N_q, 3] float32 – grid query positions for this chunk."""
        self.xyz = xyz            # [N_q, 3]
        self.centroid_num = None  # [N_latents, 3]
        self.centroid_den = None  # [N_latents]

    def __call__(self, attn_module, q: torch.Tensor, k: torch.Tensor,
                 v: torch.Tensor) -> torch.Tensor:
        # q: [B, H, N_q, d_head], k/v: [B, H, N_latents, d_head]
        scale = math.sqrt(q.shape[-1])
        # Compute attention weights explicitly (float32 for numerical stability)
        sim = torch.matmul(q.float(), k.float().transpose(-1, -2)) / scale
        # [B, H, N_q, N_latents]
        weights = sim.softmax(dim=-1)

        # Average over batch and heads → [N_q, N_latents]
        w = weights.mean(dim=(0, 1))   # [N_q, N_latents]

        xyz = self.xyz.to(w.dtype)     # [N_q, 3]
        contrib = w.T @ xyz            # [N_latents, 3]  (matmul, no huge allocation)

        if self.centroid_num is None:
            self.centroid_num = contrib
            self.centroid_den = w.sum(dim=0)
        else:
            self.centroid_num += contrib
            self.centroid_den += w.sum(dim=0)

        # Return standard attention output (same as vanilla processor)
        return torch.matmul(weights.to(v.dtype), v)


@torch.no_grad()
def compute_token_centroids(
    vae_features: torch.Tensor,        # [1, N_latents, 1024] – geo_decoder K/V
    geo_decoder,                        # ShapeVAE.geo_decoder (CrossAttentionDecoder)
    grid_res: int = 16,
    bounds: float = 1.01,
    chunk_size: int = 4096,
) -> torch.Tensor:                     # [N_latents, 3]
    """Compute attention-weighted 3D centroids for each latent token.

    Uses a coarse 3D grid of query points.  With grid_res=16 we get 17^3=4913
    queries; with chunk_size=4096 this runs in two passes and needs < 1GB VRAM.

    Returns centroids in normalised shape space, dtype float32.
    """
    import numpy as np
    try:
        from hy3dshape.hy3dshape.models.autoencoders.volume_decoders import (
            generate_dense_grid_points,
        )
    except ImportError:
        from ..models.autoencoders.volume_decoders import generate_dense_grid_points

    device = vae_features.device
    dtype = vae_features.dtype

    bbox_min = np.full(3, -bounds, dtype=np.float32)
    bbox_max = np.full(3,  bounds, dtype=np.float32)
    xyz_np, _, _ = generate_dense_grid_points(bbox_min, bbox_max, grid_res,
                                               indexing="ij")
    xyz = torch.from_numpy(xyz_np).to(device, torch.float32).reshape(-1, 3)
    N_q = xyz.shape[0]
    N_latents = vae_features.shape[1]

    acc = None  # will hold _CentroidCaptureProcessor after first chunk

    # Install capture processor once; reuse across chunks
    orig_proc = geo_decoder.cross_attn_decoder.attn.attention.attn_processor

    for start in range(0, N_q, chunk_size):
        chunk_xyz = xyz[start: start + chunk_size]  # [C, 3]
        cap = _CentroidCaptureProcessor(chunk_xyz)
        geo_decoder.cross_attn_decoder.attn.attention.attn_processor = cap

        chunk_q = chunk_xyz.to(dtype).unsqueeze(0)  # [1, C, 3]
        _ = geo_decoder(queries=chunk_q, latents=vae_features)

        if acc is None:
            acc = cap
        else:
            acc.centroid_num += cap.centroid_num
            acc.centroid_den += cap.centroid_den

    # Restore original processor
    geo_decoder.cross_attn_decoder.attn.attention.attn_processor = orig_proc

    centroids = acc.centroid_num / (acc.centroid_den.unsqueeze(1) + 1e-8)
    return centroids.cpu().float()   # [N_latents, 3]
