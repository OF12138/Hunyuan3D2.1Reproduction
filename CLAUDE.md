# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Hunyuan3D 2.1 — Tencent's open-source image-to-3D asset generation system. Two stages run independently and are composed at the application layer:

1. **Shape generation** (`hy3dshape/`) — image → untextured mesh (`.glb`). A flow-matching DiT operates in a 3DShape2VecSet latent space; a Shape VAE decodes the latent into an SDF, and a surface extractor (marching cubes etc.) converts SDF to mesh. Entry pipeline: `hy3dshape.pipelines.Hunyuan3DDiTFlowMatchingPipeline`.
2. **PBR texture generation** (`hy3dpaint/`) — mesh + reference image → textured mesh with albedo / metallic-roughness / normal maps. Multi-view diffusion renders views, results are baked back to UVs via a custom rasterizer + differentiable renderer. Entry pipeline: `textureGenPipeline.Hunyuan3DPaintPipeline` (config: `Hunyuan3DPaintConfig`).

The two stages are intentionally decoupled: shape can be used standalone, paint takes any input mesh. Top-level scripts (`demo.py`, `gradio_app.py`, `api_server.py`) wire them together by `sys.path.insert`-ing both subpackage roots before importing, because `hy3dpaint` uses unqualified imports (`from DifferentiableRenderer...`, `from utils...`) that only resolve when `hy3dpaint/` is on `sys.path`. **Preserve this pattern** when adding new entry points.

## Required Environment Setup

Tested with Python 3.10 + PyTorch 2.5.1+cu124. Two native components must be built before running anything:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Custom CUDA rasterizer (C++ extension)
cd hy3dpaint/custom_rasterizer && pip install -e . && cd ../..

# Differentiable mesh painter (compiled C/C++)
cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh && cd ../..

# Required Real-ESRGAN weight (texture pipeline expects it at this exact path)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
```

Note `hy3dpaint/custom_rasterizer/lib/` ships separate `custom_rasterizer_kernel/` and `custom_rasterizer_kernel_for_windows/` directories — Windows builds use the latter.

VRAM: ~10GB shape only, ~21GB texture only, ~29GB combined.

## Common Commands

| Task | Command |
|------|---------|
| End-to-end demo (shape + paint) | `python demo.py` |
| Gradio UI | `python gradio_app.py --model_path tencent/Hunyuan3D-2.1 --subfolder hunyuan3d-dit-v2-1 --texgen_model_path tencent/Hunyuan3D-2.1 --low_vram_mode` |
| FastAPI server | `python api_server.py` (default port 8081, see `constants.py`) |
| Exercise the API | `python test_api_server.py` (talks to `http://localhost:8081` — not a unit test suite) |
| Shape-only minimal | `cd hy3dshape && python minimal_demo.py` (or `minimal_demo_with_ckpt.py`, `minimal_vae_demo.py`) |
| Shape DiT training | `cd hy3dshape && bash scripts/train_deepspeed.sh <node_num> <node_rank> <gpus_per_node> <master_ip> <config_yaml> <output_dir>` (DeepSpeed, defaults to 8 GPUs; `train_demo.sh` is a one-node example) |
| Paint training | `cd hy3dpaint && python train.py --base cfgs/hunyuan-paint-pbr.yaml --name overfit --logdir logs/` |
| Data preprocessing (rendering / watertighting) | See `hy3dshape/tools/` (Blender 4.1 + scripts under `tools/render/` and `tools/watertight/`) |

There is no test framework, linter, or formatter configured — `test_api_server.py` is an integration probe, not pytest. Don't suggest `pytest` / `ruff` / `mypy` unless the user adds them.

## Architecture Notes That Aren't Obvious from Browsing

- **`torchvision_fix.py`** (root and `hy3dpaint/`) patches a torchvision API breakage. Every entry point imports it inside a try/except *before* importing torch-using modules. New entry points should follow the same pattern.
- **Shape pipeline** is built `diffusers`-style: `Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')`. Components live in `hy3dshape/hy3dshape/models/`: `denoisers/hunyuan3ddit.py` (DiT, with optional MoE in `moe_layers.py`), `autoencoders/` (Shape VAE + `surface_extractors.py` + `volume_decoders.py`), `diffusion/transport/` (flow-matching transport / paths / integrators).
- **Paint pipeline** orchestration in `hy3dpaint/textureGenPipeline.py` chains: `mesh_uv_wrap` → `multiviewDiffusionNet` (multi-view PBR diffusion, weights under `hunyuanpaintpbr/`) → `imageSuperNet` (Real-ESRGAN) → `ViewProcessor` baking → `MeshRender` (differentiable) → `convert_obj_to_glb` / `create_glb_with_pbr_materials`. Default: 6 views, 1024² render, 4096² texture.
- **Custom rasterizer** is a C++/CUDA extension imported as `custom_rasterizer`; its Python wrapper is `hy3dpaint/custom_rasterizer/custom_rasterizer/render.py`. The paint config has `raster_mode = "cr"` to select it.
- **API server layout**: `api_server.py` (FastAPI app, CORS, lifecycle) → `model_worker.py` (`ModelWorker` holds the loaded pipelines, runs jobs) → `api_models.py` (Pydantic request/response) → `constants.py` (defaults, OpenAPI metadata) → `logger_utils.py`. The worker runs jobs synchronously behind a semaphore for VRAM safety; tasks are tracked in-memory with UUIDs (no persistent queue). Image input is base64; output is a `.glb` file or status JSON. See `API_DOCUMENTATION.md` and `API_TESTING_SUMMARY.md` for endpoint details.
- **Training stack**: shape uses PyTorch Lightning + DeepSpeed (`hy3dshape/main.py` is the trainer entry, configs under `hy3dshape/configs/`). Paint uses its own `hy3dpaint/train.py` driven by `cfgs/hunyuan-paint-pbr.yaml`.
- **Background removal** (`hy3dshape/hy3dshape/rembg.py`) is applied automatically when input images are RGB; RGBA inputs skip it.
- **Model weights** live on HuggingFace at `tencent/Hunyuan3D-2.1` with subfolders `hunyuan3d-dit-v2-1` (shape) and `hunyuan3d-paintpbr-v2-1` (paint). The paint config additionally pulls `facebook/dinov2-giant`.

## Licensing

This repo is under the **Tencent Hunyuan Non-Commercial License**. Most source files carry a license header — preserve it when editing existing files and copy it to new files in `hy3dshape/` / `hy3dpaint/`.
