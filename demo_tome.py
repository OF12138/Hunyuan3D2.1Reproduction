"""End-to-end demo with optional ToMe-SD on Stage 1 (shape DiT).

Run baseline:
    python demo_tome.py --no-tome --out demo_baseline.glb

Run with ToMe (50% merge in middle blocks):
    python demo_tome.py --tome --ratio 0.5 --out demo_tome.glb

Compare wall-time and inspect outputs side by side.
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
from hy3dshape.tome import apply_patch, remove_patch

try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


def run(args):
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path)

    if args.tome:
        print(f"[ToMe-SD] ratio={args.ratio} skip_first={args.skip_first} skip_last={args.skip_last}")
        apply_patch(
            pipeline.model,
            ratio=args.ratio,
            skip_first=args.skip_first,
            skip_last=args.skip_last,
        )
        print(f"[ToMe-SD] patched layers: {pipeline.model._tome_config['patched_layers']}")
    else:
        print("[ToMe-SD] disabled (baseline)")

    image = Image.open(args.image).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    mesh = pipeline(image=image)[0]
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    mesh.export(args.out)
    print(f"\nDone. Wrote {args.out}")
    print(f"  wall-time:    {elapsed:.2f} s")
    print(f"  peak VRAM:    {peak_mem_gb:.2f} GB")
    print(f"  vertices:     {len(mesh.vertices)}")
    print(f"  faces:        {len(mesh.faces)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="tencent/Hunyuan3D-2.1")
    p.add_argument("--image", default="assets/demo.png")
    p.add_argument("--out", default="demo.glb")
    p.add_argument("--tome", dest="tome", action="store_true", default=True)
    p.add_argument("--no-tome", dest="tome", action="store_false")
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--skip_first", type=int, default=2)
    p.add_argument("--skip_last", type=int, default=2)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
