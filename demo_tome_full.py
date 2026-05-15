"""End-to-end demo (Stage 1 shape + Stage 2 paint) with optional ToMe-SD on Stage 1.

Stage 1 (shape) gets ToMe-SD applied to its DiT self-attention.
Stage 2 (paint) runs unchanged — included so you can inspect the final
textured GLB and confirm ToMe didn't hurt the textured result, not just
the raw geometry.

Usage:
    # baseline (no ToMe)
    python demo_tome_full.py --no-tome --tag baseline

    # with ToMe-SD at 50% merge
    python demo_tome_full.py --tome --ratio 0.5 --tag tome50

    # produces:
    #   <tag>_shape.glb          (untextured)
    #   <tag>_textured.glb       (final)
    # plus a printed timing/VRAM summary
"""
import argparse
import sys
import time

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import torch
from PIL import Image

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.tome import apply_patch

try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig


def _run_with_optional_profile(fn, profile_path=None):
    """Run fn() once, optionally under torch.profiler. Returns fn's result."""
    if not profile_path:
        return fn()

    from torch.profiler import profile, ProfilerActivity, schedule
    print(f"[profile] writing chrome trace to {profile_path}")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        result = fn()
    # Console summary: top CUDA-time kernels
    print("\n[profile] top 25 ops by CUDA time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25
    ))
    prof.export_chrome_trace(profile_path)
    return result


def stage1_shape(args, image):
    print("\n=== Stage 1: shape generation ===")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)

    if args.tome:
        print(f"[ToMe-SD] ratio={args.ratio} protect={args.protect_ratio} "
              f"skip_first={args.skip_first} skip_last={args.skip_last}")
        apply_patch(
            pipeline.model,
            ratio=args.ratio,
            skip_first=args.skip_first,
            skip_last=args.skip_last,
            protect_ratio=args.protect_ratio,
        )
        cfg = pipeline.model._tome_config
        print(f"[ToMe-SD] mode={cfg['mode']} "
              f"patched {len(cfg['patched_layers'])}/{cfg['n_blocks']} "
              f"blocks: {cfg['patched_layers']}")
    else:
        print("[ToMe-SD] disabled (baseline)")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    profile_path = (f"{args.tag}_stage1_profile.json"
                    if args.profile else None)
    mesh = _run_with_optional_profile(
        lambda: pipeline(image=image)[0],
        profile_path=profile_path,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    shape_path = f"{args.tag}_shape.glb"
    mesh.export(shape_path)

    print(f"Stage 1 done -> {shape_path}")
    print(f"  wall-time: {elapsed:.2f} s")
    print(f"  peak VRAM: {peak:.2f} GB")
    print(f"  vertices:  {len(mesh.vertices)}")
    print(f"  faces:     {len(mesh.faces)}")

    # free shape model before stage 2 (paint also needs ~21GB)
    del pipeline
    torch.cuda.empty_cache()

    return shape_path, elapsed, peak


def stage2_paint(args, shape_path):
    print("\n=== Stage 2: PBR texture generation ===")
    conf = Hunyuan3DPaintConfig(args.max_num_view, args.resolution)
    conf.realesrgan_ckpt_path = args.realesrgan_ckpt
    conf.multiview_cfg_path = args.multiview_cfg
    conf.custom_pipeline = args.custom_pipeline
    conf.multiview_pretrained_path = args.multiview_pretrained_path
    conf.dino_ckpt_path = args.dino_ckpt_path

    paint_pipeline = Hunyuan3DPaintPipeline(conf)

    out_path = f"{args.tag}_textured.glb"

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    out_path = paint_pipeline(
        mesh_path=shape_path,
        image_path=args.image,
        output_mesh_path=out_path,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"Stage 2 done -> {out_path}")
    print(f"  wall-time: {elapsed:.2f} s")
    print(f"  peak VRAM: {peak:.2f} GB")
    return out_path, elapsed, peak


def run(args):
    image = Image.open(args.image).convert("RGBA")
    if image.mode == 'RGB':
        image = BackgroundRemover()(image)

    shape_path, t1, m1 = stage1_shape(args, image)
    out_path, t2, m2 = stage2_paint(args, shape_path)

    print("\n=== Summary ===")
    print(f"  tag:           {args.tag}")
    tome_str = (f"on (ratio={args.ratio}, protect={args.protect_ratio})"
                if args.tome else "off")
    print(f"  ToMe-SD:       {tome_str}")
    print(f"  Stage 1 time:  {t1:.2f} s   peak {m1:.2f} GB")
    print(f"  Stage 2 time:  {t2:.2f} s   peak {m2:.2f} GB")
    print(f"  Total time:    {t1 + t2:.2f} s")
    print(f"  Outputs:       {shape_path}, {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="tencent/Hunyuan3D-2.1")
    p.add_argument("--image", default="assets/demo.png")
    p.add_argument("--tag", default="run", help="prefix for output files")

    # ToMe-SD switches (Stage 1 only)
    p.add_argument("--tome", dest="tome", action="store_true", default=True)
    p.add_argument("--no-tome", dest="tome", action="store_false")
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--protect_ratio", type=float, default=0.0,
                   help="fraction of tokens protected by saliency (0=vanilla, 0.3=recommended)")
    p.add_argument("--skip_first", type=int, default=2)
    p.add_argument("--skip_last", type=int, default=2)

    # Profiling
    p.add_argument("--profile", action="store_true",
                   help="run Stage 1 under torch.profiler; prints top kernels "
                        "and writes <tag>_stage1_profile.json (chrome trace)")

    # Paint config (mirrors demo.py defaults)
    p.add_argument("--max_num_view", type=int, default=6)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--realesrgan_ckpt",
                   default="hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
    p.add_argument("--multiview_cfg",
                   default="hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
    p.add_argument("--custom_pipeline",
                   default="hy3dpaint/hunyuanpaintpbr")
    p.add_argument("--multiview_pretrained_path",
                   default="/home/share/accelerate_src/openfar/models/tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1")
    p.add_argument("--dino_ckpt_path",
                   default="/home/share/accelerate_src/openfar/models/dinov2-giant")

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
