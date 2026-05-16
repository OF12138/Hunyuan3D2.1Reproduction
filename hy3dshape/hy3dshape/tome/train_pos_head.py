"""Train the PositionHead offline using VAE geo_decoder attention supervision.

Two-phase script
----------------
Phase 1 – Data generation (--generate):
    For each input image, run the shape inference pipeline to get z_0 latents,
    extract VAE features, compute attention-weighted 3D centroids, and save
    (features, centroids) pairs to disk.

Phase 2 – Training (default):
    Load saved data pairs, train Linear(1024→3) with MSE loss, report
    validation statistics, and save the head weights.

Usage
-----
# Step 1: generate data from images (slow – runs inference once per image)
python -m hy3dshape.hy3dshape.tome.train_pos_head \\
    --generate \\
    --images_dir   assets/train_images \\
    --data_dir     tome_data/centroids \\
    --model_path   tencent/Hunyuan3D-2.1 \\
    --grid_res     16

# Step 2: train the head on the generated data
python -m hy3dshape.hy3dshape.tome.train_pos_head \\
    --data_dir     tome_data/centroids \\
    --output       tome_data/pos_head.pt \\
    --epochs       50 \\
    --lr           1e-3 \\
    --val_split    0.2

# To load and use at inference time:
#   from hy3dshape.hy3dshape.tome import apply_patch
#   apply_patch(pipeline.model, ratio=0.5,
#               pos_head_path="tome_data/pos_head.pt")
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow running either as `python -m hy3dshape.hy3dshape.tome.train_pos_head`
# or as `python train_pos_head.py` from within the tome directory.
# parents[3] = repo root (e.g. .../Hunyuan), parents[2] = .../Hunyuan/hy3dshape
_REPO = Path(__file__).resolve().parents[3]
_PKGROOT = _REPO / "hy3dshape"
# Put _PKGROOT FIRST so that `import hy3dshape` resolves to the regular package
# at hy3dshape/hy3dshape/, not the outer namespace dir hy3dshape/.
for _p in [str(_REPO), str(_PKGROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# If the outer namespace-package version of `hy3dshape` was imported by `-m`,
# evict it so the next `import hy3dshape` picks up the real package on _PKGROOT.
_cached = sys.modules.get("hy3dshape")
if _cached is not None and not hasattr(_cached, "pipelines"):
    for _k in [k for k in sys.modules if k == "hy3dshape" or k.startswith("hy3dshape.")]:
        # Keep our own already-loaded sibling modules; only drop the bare/outer one.
        if _k == "hy3dshape":
            del sys.modules[_k]

try:
    from .pos_head import PositionHead, compute_token_centroids
except ImportError:
    from pos_head import PositionHead, compute_token_centroids  # script mode


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------

def _load_pipeline(model_path: str, device: str = "cuda"):
    """Load shape pipeline + ShapeVAE.  Returns (pipeline, vae)."""
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    pipeline.to(device)
    # vae is the ShapeVAE attached to the pipeline
    vae = pipeline.vae
    return pipeline, vae


def _run_inference_capture(pipeline, image, capture_block_indices, device: str = "cuda"):
    """Run a full inference pass and capture two things at the **last** denoising step:

    1. DiT block-input hidden states for each block in `capture_block_indices`,
       shape [B_cfg, 1+N_latents, D_dit]   (D_dit = 2048 for Hunyuan3D-2.1).
       The +1 leading token is the timestep/conditioning CLS – we strip it later.
    2. The post-transformer VAE features [1, N_latents, D_vae] arriving at
       `vae.latents2mesh`, used downstream for VAE-attention-based centroids.

    Returns (block_features_dict, vae_features).  block_features_dict maps
    block_idx -> tensor on CPU.
    """
    captured_blocks = {}  # block_idx -> latest input tensor
    captured_feats = {}

    # Forward-pre-hooks: on each model.forward call, overwrite the slot, so
    # at the end of inference each slot holds the LAST denoising step's input.
    hooks = []
    for idx in capture_block_indices:
        block = pipeline.model.blocks[idx]

        def make_hook(bi):
            def _pre_hook(module, args, kwargs):
                # signature: forward(self, x, c=None, text_states=None, skip_value=None)
                x = args[0]
                captured_blocks[bi] = x.detach()
            return _pre_hook

        hooks.append(block.register_forward_pre_hook(
            make_hook(idx), with_kwargs=True))

    _orig = pipeline.vae.latents2mesh

    def _vae_hook(latents, **kw):
        captured_feats["feats"] = latents.detach()
        return _orig(latents, **kw)

    pipeline.vae.latents2mesh = _vae_hook
    try:
        pipeline(image=image)
    finally:
        for h in hooks:
            h.remove()
        pipeline.vae.latents2mesh = _orig

    return captured_blocks, captured_feats.get("feats")


def generate_data(args):
    """Phase 1: images → (DiT block features, VAE-derived centroids) pairs.

    For each image we run a full inference pass and, at the LAST denoising step,
    capture the input hidden state to each DiT block in --capture_blocks.  We
    pair those features with centroids derived from the VAE geo_decoder's
    cross-attention on the final clean latent.  One .pt file per (image, block).
    """
    import glob
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline, vae = _load_pipeline(args.model_path, device)
    dtype = next(vae.parameters()).dtype

    image_files = sorted(
        glob.glob(str(Path(args.images_dir) / "**" / "*"), recursive=True)
    )
    image_files = [f for f in image_files
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    if args.max_images is not None and args.max_images > 0:
        image_files = image_files[: args.max_images]

    if not image_files:
        print(f"No images found in {args.images_dir}")
        return

    capture_blocks = [int(s) for s in args.capture_blocks.split(",")]
    print(f"Found {len(image_files)} images. "
          f"Capturing DiT inputs at blocks {capture_blocks}.")
    print(f"Writing data to {out_dir} ...")

    for idx, img_path in enumerate(image_files):
        stem = Path(img_path).stem
        all_exist = all(
            (out_dir / f"{stem}_{idx:04d}_b{b:02d}.pt").exists()
            for b in capture_blocks
        )
        if all_exist:
            print(f"  [{idx+1}/{len(image_files)}] skip {stem} (already exists)")
            continue

        print(f"  [{idx+1}/{len(image_files)}] {stem}")
        try:
            image = Image.open(img_path).convert("RGBA")

            t0 = time.time()
            block_feats, vae_feats = _run_inference_capture(
                pipeline, image, capture_blocks, device,
            )
            if vae_feats is None or not block_feats:
                print(f"    WARNING: capture failed, skipping.")
                continue

            # Centroids from VAE geo_decoder (one set per image)
            vae_feats = vae_feats.to(device, dtype)
            centroids = compute_token_centroids(
                vae_feats, vae.geo_decoder,
                grid_res=args.grid_res,
                bounds=1.01,
            )   # [N_latents, 3]

            for b_idx, x_cfg in block_feats.items():
                # x_cfg: [B_cfg, 1+N_latents, D_dit].  Take cond half, drop CLS.
                x = x_cfg[0, 1:, :].contiguous()    # [N_latents, D_dit]
                out_path = out_dir / f"{stem}_{idx:04d}_b{b_idx:02d}.pt"
                torch.save({
                    "features": x.cpu().half(),         # [N, D_dit]
                    "centroids": centroids.float(),     # [N, 3]
                    "block_idx": b_idx,
                }, out_path)

            sample_feat = next(iter(block_feats.values()))
            print(f"    DiT feats {tuple(sample_feat.shape)} × {len(block_feats)} blocks, "
                  f"centroids range [{centroids.min():.3f}, {centroids.max():.3f}] "
                  f"({time.time()-t0:.1f}s)")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    print("Data generation done.")


# ---------------------------------------------------------------------------
# Phase 2: training
# ---------------------------------------------------------------------------

class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = torch.load(self.paths[idx], weights_only=True)
        feats = d["features"].float()      # [N, D_dit]
        centroids = d["centroids"].float() # [N, 3]
        return feats, centroids


def train(args):
    """Phase 2: train PositionHead on saved (features, centroids) pairs."""
    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*.pt"))
    if not all_files:
        print(f"No .pt files found in {data_dir}.  Run with --generate first.")
        return

    print(f"Found {len(all_files)} data files.")
    random.shuffle(all_files)
    n_val = max(1, int(len(all_files) * args.val_split))
    val_files  = all_files[:n_val]
    train_files = all_files[n_val:]
    print(f"  train={len(train_files)}, val={len(val_files)}")

    train_ds = _PairDataset(train_files)
    val_ds   = _PairDataset(val_files)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Auto-detect feature dim from first data file
    probe = torch.load(all_files[0], weights_only=True)
    hidden_size = int(probe["features"].shape[-1])
    print(f"  hidden_size (auto) = {hidden_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = PositionHead(hidden_size=hidden_size).to(device)
    optimizer = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_loss = float("inf")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        head.train()
        total_loss = 0.0
        n_samples = 0
        for feats, centroids in train_loader:
            # feats: [B, N, 1024], centroids: [B, N, 3]
            feats = feats.to(device)
            centroids = centroids.to(device)

            pred = head(feats)   # [B, N, 3]
            loss = nn.functional.mse_loss(pred, centroids)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * feats.shape[0]
            n_samples  += feats.shape[0]

        scheduler.step()
        train_loss = total_loss / max(n_samples, 1)

        # --- validate ---
        head.eval()
        val_loss = 0.0
        l2_errors = []
        with torch.no_grad():
            for feats, centroids in val_loader:
                feats = feats.to(device)
                centroids = centroids.to(device)
                pred = head(feats)
                val_loss += nn.functional.mse_loss(pred, centroids).item() * feats.shape[0]
                err = (pred - centroids).norm(dim=-1)  # [B, N]
                l2_errors.append(err.cpu())

        val_loss /= max(len(val_ds), 1)
        l2_err = torch.cat([e.reshape(-1) for e in l2_errors])
        p50 = l2_err.quantile(0.50).item()
        p95 = l2_err.quantile(0.95).item()
        mean_err = l2_err.mean().item()

        print(f"[epoch {epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
              f"L2_mean={mean_err:.4f}  p50={p50:.4f}  p95={p95:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            head.save(str(out_path))
            print(f"  => saved best head to {out_path}")

    print(f"\nTraining done.  Best val_loss={best_val_loss:.5f}  "
          f"Head saved to {out_path}")


# ---------------------------------------------------------------------------
# Validation-only mode: load a saved head and report error statistics
# ---------------------------------------------------------------------------

def validate(args):
    """Load a trained head and report per-token centroid error statistics."""
    from pathlib import Path

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.pt"))
    if not files:
        print("No data files found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = PositionHead.load(args.output, device=device)
    head.to(device)

    all_errors = []
    for p in files:
        d = torch.load(p, weights_only=True)
        feats = d["features"].float().unsqueeze(0).to(device)   # [1, N, 1024]
        centroids = d["centroids"].float().unsqueeze(0).to(device)  # [1, N, 3]
        with torch.no_grad():
            pred = head(feats)
        err = (pred - centroids).norm(dim=-1).squeeze(0).cpu()  # [N]
        all_errors.append(err)

    all_errors = torch.cat(all_errors)  # [total_tokens]
    print(f"Validation on {len(files)} samples ({all_errors.shape[0]} tokens):")
    print(f"  mean L2 error : {all_errors.mean():.4f}")
    print(f"  p25           : {all_errors.quantile(0.25):.4f}")
    print(f"  p50 (median)  : {all_errors.quantile(0.50):.4f}")
    print(f"  p75           : {all_errors.quantile(0.75):.4f}")
    print(f"  p95           : {all_errors.quantile(0.95):.4f}")
    print(f"  max           : {all_errors.max():.4f}")

    # Spatial coverage: how well do centroids cover [-1,1]^3?
    all_preds = []
    for p in files:
        d = torch.load(p, weights_only=True)
        feats = d["features"].float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = head(feats).squeeze(0).cpu()
        all_preds.append(pred)
    preds = torch.cat(all_preds, dim=0)  # [total_tokens, 3]
    print(f"\nPredicted centroid range per axis:")
    for i, axis in enumerate("xyz"):
        print(f"  {axis}: [{preds[:, i].min():.3f}, {preds[:, i].max():.3f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Train / validate the PositionHead for locality-aware ToMe.")
    p.add_argument("--data_dir", default="tome_data/centroids",
                   help="Directory for saved (features, centroids) .pt files.")
    p.add_argument("--output", default="tome_data/pos_head.pt",
                   help="Where to save / load the trained head.")

    # Phase 1
    p.add_argument("--generate", action="store_true",
                   help="Run data generation phase (needs --images_dir).")
    p.add_argument("--images_dir", default="assets",
                   help="Directory containing input images for data gen.")
    p.add_argument("--model_path", default="tencent/Hunyuan3D-2.1",
                   help="Path or HuggingFace ID for the shape model.")
    p.add_argument("--grid_res", type=int, default=16,
                   help="Coarse grid resolution for centroid extraction (17^3 ≈ 5k pts).")
    p.add_argument("--max_images", type=int, default=None,
                   help="If set, only process the first N images from --images_dir.")
    p.add_argument("--capture_blocks", default="2,8,14,18",
                   help="Comma-separated DiT block indices to capture DiT hidden "
                        "states from (training mixes across these for generality).")

    # Phase 2
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Number of latent samples per batch (each has 4096 tokens).")
    p.add_argument("--val_split", type=float, default=0.2)

    # Validation-only
    p.add_argument("--validate_only", action="store_true",
                   help="Skip training; just evaluate the saved head.")

    args = p.parse_args()

    if args.generate:
        generate_data(args)
    elif args.validate_only:
        validate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
