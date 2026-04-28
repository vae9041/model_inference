#!/usr/bin/env python3
"""
Run Faster R-CNN (ResNet-18) inference on images from a file or directory and benchmark latency.

For standard training checkpoints (e.g. `final_model.pth` with `model_state_dict`).
For structured-pruned saves, use `inference_structured_pruning_resnet_18.py` instead.

Input matches training / eval: list of float CHW tensors in [0, 1], resized to --width x --height.

Examples:
  python inference_resnet_18.py \\
    --checkpoint ./models_resnet18_v2/final_model.pth \\
    --image_dir ./images

  python inference_resnet_18.py \\
    --checkpoint ./models_resnet18_v2/final_model.pth \\
    --image_path ./images/test.png \\
    --runs 50 --warmup 10 --fp16
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List

import cv2
import torch

from train_resnet_18 import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def collect_image_paths(image_path: Path | None, image_dir: Path | None, recursive: bool) -> List[Path]:
    if image_path is not None:
        if not image_path.is_file():
            raise FileNotFoundError(f"Not a file: {image_path}")
        return [image_path]
    if image_dir is None:
        raise ValueError("Provide --image_path or --image_dir")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {image_dir}")
    if recursive:
        paths = sorted(
            p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )
    else:
        paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
        )
    if not paths:
        raise FileNotFoundError(f"No images with extensions {IMAGE_EXTENSIONS} under {image_dir}")
    return paths


def preprocess_bgr_to_chw(
    bgr: "cv2.Mat",
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """RGB resize -> (3, H, W) float [0,1] on device."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=dtype, device=device)
    t = t / 255.0
    return t


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = get_model(num_classes=2, freeze_backbone=False)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded model_state_dict"
            + (f" (epoch {ckpt.get('epoch', '?')})" if "epoch" in ckpt else "")
        )
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Inference + latency benchmark for ResNet-18 Faster R-CNN")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="./models_resnet18_v2/final_model.pth",
        help="Path to .pth (e.g. final_model.pth)",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image_path", type=str, default=None, help="Single image file")
    g.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    p.add_argument("--recursive", action="store_true", help="With --image_dir, scan subfolders")
    p.add_argument("--width", type=int, default=640, help="Resize width (match training)")
    p.add_argument("--height", type=int, default=480, help="Resize height (match training)")
    p.add_argument("--device", type=str, default="auto", help="cuda / cpu / auto")
    p.add_argument("--warmup", type=int, default=10, help="Warmup forwards before timing")
    p.add_argument("--runs", type=int, default=30, help="Timed forwards per image (after warmup on first image)")
    p.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA (faster on many GPUs)")
    p.add_argument("--include_preprocess", action="store_true", help="Include disk read + resize in timed loop")
    p.add_argument("--conf_threshold", type=float, default=0.5, help="Log count of boxes with score >= this")
    p.add_argument("--save_dir", type=str, default=None, help="Optional folder to save annotated images")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float32
    if args.fp16:
        if device.type != "cuda":
            logger.warning("--fp16 ignored (CUDA not used)")
        else:
            dtype = torch.float16

    ckpt_path = Path(args.checkpoint)
    paths = collect_image_paths(
        Path(args.image_path) if args.image_path else None,
        Path(args.image_dir) if args.image_dir else None,
        args.recursive,
    )

    logger.info(f"Device: {device}, dtype: {dtype}, images: {len(paths)}")
    model = load_model(ckpt_path, device)
    if args.fp16 and device.type == "cuda":
        model.half()

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    first_bgr = cv2.imread(str(paths[0]))
    if first_bgr is None:
        raise RuntimeError(f"Could not read image: {paths[0]}")
    if not args.include_preprocess:
        w_tensor = preprocess_bgr_to_chw(first_bgr, args.width, args.height, device, dtype)
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            if args.include_preprocess:
                bgr = cv2.imread(str(paths[0]))
                w_tensor = preprocess_bgr_to_chw(bgr, args.width, args.height, device, dtype)
            _ = model([w_tensor])
    if device.type == "cuda":
        torch.cuda.synchronize()

    all_ms: List[float] = []
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            logger.warning(f"Skip unreadable: {path}")
            continue
        if not args.include_preprocess:
            inp = preprocess_bgr_to_chw(bgr, args.width, args.height, device, dtype)
        times: List[float] = []
        with torch.inference_mode():
            for _ in range(max(1, args.runs)):
                if args.include_preprocess:
                    bgr2 = cv2.imread(str(path))
                    inp = preprocess_bgr_to_chw(bgr2, args.width, args.height, device, dtype)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model([inp])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = sum(times) / len(times)
        all_ms.append(mean_ms)

        pred = out[0]
        n_keep = int((pred["scores"] >= args.conf_threshold).sum().item())
        logger.info(
            f"{path.name}: mean {mean_ms:.2f} ms / forward (n={len(times)}), "
            f"boxes>={args.conf_threshold}: {n_keep} (raw {pred['boxes'].shape[0]})"
        )

        if save_dir is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_r = cv2.resize(rgb, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            vis = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR)
            boxes = pred["boxes"].detach().float().cpu().numpy()
            scores = pred["scores"].detach().float().cpu().numpy()
            for box, sc in zip(boxes, scores):
                if sc < args.conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis, f"{sc:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )
            out_p = save_dir / f"pred_{path.stem}.png"
            cv2.imwrite(str(out_p), vis)

    if all_ms:
        overall = sum(all_ms) / len(all_ms)
        print("\n" + "=" * 60)
        print("INFERENCE LATENCY (model forward" + (", incl. preprocess" if args.include_preprocess else "") + ")")
        print("=" * 60)
        print(f"Checkpoint     : {ckpt_path}")
        print(f"Resize         : {args.width}x{args.height}")
        print(f"FP16           : {args.fp16 and device.type == 'cuda'}")
        print(f"Images         : {len(all_ms)}")
        print(f"Runs / image   : {args.runs}")
        print(f"Mean per image : {overall:.2f} ms")
        print(f"Min / Max      : {min(all_ms):.2f} ms / {max(all_ms):.2f} ms")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
