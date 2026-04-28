#!/usr/bin/env python3
"""
Structured (channel) pruning for Faster R-CNN with ResNet-18 FPN (same architecture as train_resnet_18).

Uses Torch-Pruning (dependency-aware channel removal) so pruned tensors are actually smaller than
unstructured zeros — this is what can reduce FLOPs and latency on GPU.

Dependency:
    pip install 'torch-pruning>=1.3'

About the 8 ms goal:
    End-to-end Faster R-CNN at 640×480 rarely reaches <8 ms in pure PyTorch FP32 on most GPUs.
    Structured pruning helps, but sub-8 ms usually also needs: smaller input resolution, FP16
    (half()), torch.compile (PyTorch 2+), and/or TensorRT / ONNX Runtime. This script prints a
    latency benchmark (FP32 / FP16 / optional compile) so you can see what actually moves the needle.

Workflow:
    1) Load models_resnet18_v2/final_model.pth into get_model()
    2) Replace FrozenBatchNorm2d → BatchNorm2d (Torch-Pruning aligns channels with regular BN)
    3) Structured prune (default: FPN 3×3 towers only — minutes; ResNet + 1×1 laterals unchanged). Full-model prune can take many hours.
    4) After FPN prune: enforce one channel width (1×1 projectors if needed), then rebuild RPN + ROI heads (fine-tune)
    5) Optional fine-tune + save checkpoint + benchmark

Example:
    python structured_pruning_resnet_18.py \\
      --checkpoint ./models_resnet18_v2/final_model.pth \\
      --output_dir ./models_resnet18_v2_structured \\
      --pruning_ratio 0.35 \\
      --prune_iterations 8 \\
      --example_height 480 --example_width 640 \\
      --finetune_epochs 5 --lr 1e-4 \\
      --benchmark --benchmark_compile
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

def _get_torch_pruning():
    """Lazy import so inference-only code paths (e.g. load_structured_pruned_state_dict) work without torch-pruning."""
    try:
        import torch_pruning as tp
    except ImportError:
        print(
            "Missing torch-pruning. Install with:\n  pip install 'torch-pruning>=1.3'",
            file=sys.stderr,
        )
        sys.exit(1)
    return tp

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops.misc import FrozenBatchNorm2d

from train import GraspDataset, collate_fn, train_one_epoch
from train_resnet_18 import evaluate_with_diagnostics, get_model, print_model_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _heartbeat_loop(stop: threading.Event, label: str, interval_sec: float) -> None:
    """Daemon thread: log every interval until stop is set (proves main thread is still alive)."""
    t0 = time.perf_counter()
    while not stop.wait(timeout=interval_sec):
        elapsed = time.perf_counter() - t0
        logger.info(
            f"{label}: still working… {elapsed:.0f}s elapsed. "
            "This phase is one big CPU/GPU job, not a Python loop. "
            "If GPU use shows in nvidia-smi, it is not frozen."
        )


def replace_frozen_batchnorm_with_batchnorm2d(module: nn.Module) -> int:
    """
    Torch-Pruning tracks channel dependencies through nn.BatchNorm2d; FrozenBatchNorm2d often
    breaks that linkage. Replace in-place; returns number of modules replaced.
    """
    replaced = 0

    def recurse(parent: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            if isinstance(child, FrozenBatchNorm2d):
                num = int(child.weight.shape[0])
                dev, dt = child.weight.device, child.weight.dtype
                bn = nn.BatchNorm2d(num, eps=child.eps, affine=True, track_running_stats=True)
                bn.to(device=dev, dtype=dt)
                bn.weight.data.copy_(child.weight)
                bn.bias.data.copy_(child.bias)
                bn.running_mean.data.copy_(child.running_mean)
                bn.running_var.data.copy_(child.running_var)
                bn.eval()
                setattr(parent, name, bn)
                replaced += 1
            else:
                recurse(child)

    recurse(module)
    return replaced


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_pruning_parameter_report(total_before: int, total_after: int) -> None:
    """Print an easy-to-scan parameter delta report."""
    delta = total_after - total_before
    pct = 0.0 if total_before == 0 else (delta / total_before) * 100.0
    print("\n" + "=" * 64)
    print("STRUCTURED PRUNING PARAMETER REPORT")
    print("=" * 64)
    print(f"Total parameters before pruning : {total_before:,}")
    print(f"Total parameters after pruning  : {total_after:,}")
    print(f"Parameter delta                : {delta:+,} ({pct:+.2f}%)")
    print("=" * 64 + "\n")


def benchmark_forward(
    model: nn.Module,
    device: torch.device,
    height: int,
    width: int,
    warmup: int,
    runs: int,
    use_fp16: bool,
    use_compile: bool,
) -> float:
    """Mean seconds per image (list batch of 1) for inference-only forward."""
    model.eval()
    if use_fp16 and device.type != "cuda":
        logger.warning("FP16 benchmark skipped (CUDA not used).")
        use_fp16 = False

    dtype = torch.float16 if use_fp16 else torch.float32
    dummy = [torch.randn(3, height, width, device=device, dtype=dtype)]

    m: nn.Module = model
    if use_fp16:
        model.half()
    if use_compile and hasattr(torch, "compile"):
        try:
            m = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}; benchmarking eager model.")
            m = model
    elif use_compile:
        logger.warning("torch.compile not available in this PyTorch build.")

    try:
        with torch.no_grad():
            for _ in range(warmup):
                m(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times: List[float] = []
        with torch.no_grad():
            for _ in range(runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                m(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        return sum(times) / len(times)
    finally:
        if use_fp16:
            model.float()


def run_structured_prune(
    model: nn.Module,
    device: torch.device,
    example_inputs: list[torch.Tensor],
    pruning_ratio: float,
    prune_iterations: int,
    round_to: Optional[int],
    isomorphic: bool,
    graph_heartbeat_sec: float,
) -> None:
    tp = _get_torch_pruning()
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
    ignored: List[nn.Module] = [model.transform]

    logger.info(
        "Building Torch-Pruning dependency graph for the **entire** Faster R-CNN (RPN + ROI + backbone). "
        "This routinely takes **hours** (you saw ~2h+ still building). Prefer --prune_scope backbone unless you "
        "really need full-graph pruning."
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    stop_hb = threading.Event()
    hb_thread: Optional[threading.Thread] = None
    if graph_heartbeat_sec > 0:
        hb_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(stop_hb, "Dependency graph build", graph_heartbeat_sec),
            daemon=True,
            name="tp-graph-heartbeat",
        )
        hb_thread.start()
    try:
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_inputs,
            importance=imp,
            pruning_ratio=pruning_ratio,
            global_pruning=True,
            isomorphic=isomorphic,
            iterative_steps=prune_iterations,
            ignored_layers=ignored,
            round_to=round_to,
            # Faster R-CNN expects images: list[Tensor], not *args unpacking.
            forward_fn=lambda mod, inp: mod(inp),
        )
    finally:
        stop_hb.set()
    logger.info(
        f"Dependency graph built in {time.perf_counter() - t0:.1f}s; running {prune_iterations} prune steps…"
    )
    for s in range(prune_iterations):
        t_step = time.perf_counter()
        pruner.step()
        logger.info(
            f"Structured prune step {s + 1}/{prune_iterations} done ({time.perf_counter() - t_step:.1f}s)."
        )


class _BackboneWithUniformFPN(nn.Module):
    """
    Wraps torchvision BackboneWithFPN (body + fpn) and applies a 1×1 per level so every map has
    the same C (RPN requirement). Survives torch.save/load via state_dict 
    """

    def __init__(self, body: nn.Module, fpn: nn.Module, projectors: nn.ModuleDict, out_ch: int):
        super().__init__()
        self.body = body
        self.fpn = fpn
        self.fpn_output_projectors = projectors
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> OrderedDict:
        feats = self.fpn(self.body(x))
        return OrderedDict((k, self.fpn_output_projectors[k](v)) for k, v in feats.items())


def measure_backbone_out_channels(model: nn.Module, device: torch.device, height: int, width: int) -> int:
    """FPN outputs must share one channel width; return it."""
    dtype = next(model.parameters()).dtype
    x = torch.randn(1, 3, height, width, device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        feats = model.backbone(x)
    chans = [t.shape[1] for t in feats.values()]
    u = set(chans)
    if len(u) != 1:
        raise RuntimeError(f"Backbone FPN levels disagree on C: {chans}")
    return chans[0]


def ensure_uniform_fpn_backbone(
    model: nn.Module,
    device: torch.device,
    height: int,
    width: int,
    target_channels: Optional[int],
) -> tuple[int, Optional[dict]]:
    """
    Torch-Pruning often leaves FPN pyramid levels with different C; RPN needs one width.
    If needed, replace `model.backbone` with a wrapper + learned 1×1 projectors (Kaiming init).

    Returns:
        (uniform_channels, structured_meta or None). Save structured_meta in the checkpoint so
        load_structured_pruned_state_dict can rebuild the wrapper before loading weights.
    """
    backbone = model.backbone
    dtype = next(backbone.parameters()).dtype
    x = torch.randn(1, 3, height, width, device=device, dtype=dtype)
    backbone.eval()
    with torch.no_grad():
        feats = backbone(x)
    keys = list(feats.keys())
    chans = [feats[k].shape[1] for k in keys]
    if len(set(chans)) == 1:
        backbone.out_channels = chans[0]
        return chans[0], None

    if isinstance(backbone, _BackboneWithUniformFPN):
        raise RuntimeError("Backbone already wrapped; internal FPN width mismatch should not happen.")

    if target_channels is not None:
        uniform_c = int(target_channels)
    else:
        mx = max(chans)
        floor = max(32, mx // 4)
        big = [c for c in chans if c >= floor]
        uniform_c = max(big) if big else mx

    projs = nn.ModuleDict()
    for k, c in zip(keys, chans):
        if c == uniform_c:
            projs[k] = nn.Identity()
        else:
            layer = nn.Conv2d(c, uniform_c, kernel_size=1, bias=False)
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            projs[k] = layer.to(device=device, dtype=dtype)

    body, fpn = backbone.body, backbone.fpn
    wrapped = _BackboneWithUniformFPN(body, fpn, projs, uniform_c).to(device=device, dtype=dtype)
    model.backbone = wrapped
    logger.info(
        "Torch-Pruning produced unequal FPN widths %s; wrapped backbone with 1×1 projectors → %d ch.",
        list(zip(keys, chans)),
        uniform_c,
    )
    meta = {
        "fpn_uniform_wrapper": True,
        "uniform_c": uniform_c,
        "fpn_levels": [{"key": k, "in_ch": c} for k, c in zip(keys, chans)],
    }
    return uniform_c, meta


def load_structured_pruned_state_dict(
    model: nn.Module,
    checkpoint: dict,
    device: torch.device,
) -> None:
    """
    Load a checkpoint saved by this script. Rebuilds `_BackboneWithUniformFPN` when metadata says so.
    """
    meta = checkpoint.get("structured_pruning_meta")
    sd = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    if not isinstance(sd, dict):
        raise TypeError("checkpoint must contain model_state_dict or be a state dict")

    if meta and meta.get("fpn_uniform_wrapper"):
        bb = model.backbone
        if isinstance(bb, _BackboneWithUniformFPN):
            model.load_state_dict(sd, strict=True)
            return
        dtype = next(bb.parameters()).dtype
        uniform_c = int(meta["uniform_c"])
        projs = nn.ModuleDict()
        for lev in meta["fpn_levels"]:
            k, cin = lev["key"], int(lev["in_ch"])
            if cin == uniform_c:
                projs[k] = nn.Identity()
            else:
                layer = nn.Conv2d(cin, uniform_c, kernel_size=1, bias=False)
                projs[k] = layer.to(device=device, dtype=dtype)
        model.backbone = _BackboneWithUniformFPN(bb.body, bb.fpn, projs, uniform_c).to(
            device=device, dtype=dtype
        )

    # Rebuild pruned FPN layer_blocks to checkpoint shapes (they are no longer all 256x256).
    # Example keys:
    #   backbone.fpn.layer_blocks.0.0.weight -> [200, 256, 3, 3]
    fpn = model.backbone.fpn
    for i in range(len(fpn.layer_blocks)):
        w_key = f"backbone.fpn.layer_blocks.{i}.0.weight"
        b_key = f"backbone.fpn.layer_blocks.{i}.0.bias"
        if w_key not in sd:
            continue
        w = sd[w_key]
        if w.ndim != 4:
            continue
        out_ch, in_ch, k_h, k_w = w.shape
        has_bias = b_key in sd
        conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(k_h, k_w),
            stride=1,
            padding=((k_h - 1) // 2, (k_w - 1) // 2),
            bias=has_bias,
        ).to(device=device, dtype=w.dtype)
        fpn.layer_blocks[i] = nn.Sequential(conv)

    # Rebuild detector heads to checkpoint width (RPN + ROI). This handles the 224-channel
    # resized heads used after structured pruning.
    if "rpn.head.conv.0.0.weight" in sd and "rpn.head.cls_logits.weight" in sd:
        c = int(sd["rpn.head.conv.0.0.weight"].shape[0])
        n_anchors = int(sd["rpn.head.cls_logits.weight"].shape[0])
        model.rpn.head = RPNHead(c, n_anchors).to(device=device, dtype=sd["rpn.head.conv.0.0.weight"].dtype)

    if "roi_heads.box_head.fc6.weight" in sd and "roi_heads.box_predictor.cls_score.weight" in sd:
        fc6_in = int(sd["roi_heads.box_head.fc6.weight"].shape[1])
        rep = int(sd["roi_heads.box_head.fc6.weight"].shape[0])
        num_classes = int(sd["roi_heads.box_predictor.cls_score.weight"].shape[0])
        model.roi_heads.box_head = TwoMLPHead(fc6_in, rep).to(
            device=device, dtype=sd["roi_heads.box_head.fc6.weight"].dtype
        )
        model.roi_heads.box_predictor = FastRCNNPredictor(rep, num_classes).to(
            device=device, dtype=sd["roi_heads.box_predictor.cls_score.weight"].dtype
        )

    # Keep Faster R-CNN bookkeeping consistent with reconstructed width.
    if meta and meta.get("fpn_uniform_wrapper"):
        model.backbone.out_channels = int(meta["uniform_c"])

    model.load_state_dict(sd, strict=True)


def rebuild_detector_heads_for_backbone_width(model: nn.Module, num_classes: int, backbone_out_ch: int) -> None:
    """
    After channel pruning on `model.backbone`, RPN and ROI MLPs must match the new FPN width.
    New head weights are randomly initialized (same as fresh Faster R-CNN heads).
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.backbone.out_channels = backbone_out_ch

    n_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
    model.rpn.head = RPNHead(backbone_out_ch, n_anchors).to(device=device, dtype=dtype)

    resolution = model.roi_heads.box_roi_pool.output_size[0]
    if not isinstance(model.roi_heads.box_head, TwoMLPHead):
        raise TypeError("Expected torchvision TwoMLPHead on roi_heads.box_head; edit script for custom heads.")
    representation_size = model.roi_heads.box_head.fc7.out_features
    model.roi_heads.box_head = TwoMLPHead(
        backbone_out_ch * resolution**2, representation_size
    ).to(device=device, dtype=dtype)
    model.roi_heads.box_predictor = FastRCNNPredictor(representation_size, num_classes).to(
        device=device, dtype=dtype
    )


def run_structured_prune_fpn_only(
    model: nn.Module,
    device: torch.device,
    example_nchw: torch.Tensor,
    pruning_ratio: float,
    prune_iterations: int,
    round_to: Optional[int],
    isomorphic: bool,
    graph_heartbeat_sec: float,
) -> None:
    """
    Prune only the FPN 3×3 `layer_blocks` (merged features are already a fixed width; 1×1 laterals stay).

    Pruning the full BackboneWithFPN (ResNet + FPN) breaks ResNet residual/BN coupling; pruning both
    inner and layer FPN blocks in one mixed global scope breaks equal pyramid width for RPN.
    """
    tp = _get_torch_pruning()
    fpn = model.backbone.fpn
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")

    # RPN needs identical C on every pyramid map (including the extra "pool" level).
    # Only the four 3×3 `layer_blocks` see the same tensor width (merged 256-d features);
    # tying `inner_blocks` into the same global scope mixes 64/128/256/512→256 1×1 convs and
    # breaks equal layer_block output widths (e.g. [192,200,208,16,16]).
    fpn_layer_only: tuple[nn.Module, ...] = tuple(fpn.layer_blocks)
    pruning_ratio_dict = {fpn_layer_only: pruning_ratio}

    def forward_fn(fpn_mod: nn.Module, inp: torch.Tensor):
        feats = model.backbone.body(inp)
        return fpn_mod(feats)

    logger.info(
        "Building Torch-Pruning graph for **FPN 3×3 layer_blocks only**. "
        "If pyramid widths still differ afterward, the script adds 1×1 projectors for a uniform RPN width."
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    stop_hb = threading.Event()
    if graph_heartbeat_sec > 0:
        threading.Thread(
            target=_heartbeat_loop,
            args=(stop_hb, "FPN dependency graph build", graph_heartbeat_sec),
            daemon=True,
            name="tp-fpn-heartbeat",
        ).start()
    try:
        pruner = tp.pruner.MagnitudePruner(
            fpn,
            example_inputs=example_nchw,
            importance=imp,
            pruning_ratio=0.0,
            pruning_ratio_dict=pruning_ratio_dict,
            global_pruning=True,
            isomorphic=isomorphic,
            iterative_steps=prune_iterations,
            ignored_layers=None,
            round_to=round_to,
            forward_fn=forward_fn,
        )
    finally:
        stop_hb.set()

    logger.info(
        f"FPN graph built in {time.perf_counter() - t0:.1f}s; running {prune_iterations} prune steps…"
    )
    for s in range(prune_iterations):
        t_step = time.perf_counter()
        pruner.step()
        logger.info(
            f"FPN prune step {s + 1}/{prune_iterations} done ({time.perf_counter() - t_step:.1f}s)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured prune ResNet-18 Faster R-CNN")
    parser.add_argument("--checkpoint", type=str, default="./models_resnet18_v2/final_model.pth")
    parser.add_argument("--output_dir", type=str, default="./models_resnet18_v2_structured")

    parser.add_argument("--pruning_ratio", type=float, default=0.35, help="Target channel sparsity (approx., global)")
    parser.add_argument(
        "--prune_iterations",
        type=int,
        default=8,
        help="Split pruning into this many steps (smaller steps = stabler ResNet-FPN)",
    )
    parser.add_argument("--round_to", type=int, default=8, help="Round channels to multiple of this (GPU-friendly)")
    parser.add_argument("--no_isomorphic", action="store_true", help="Disable isomorphic global pruning (not recommended)")
    parser.add_argument(
        "--prune_scope",
        type=str,
        default="backbone",
        choices=["backbone", "full"],
        help="backbone: prune FPN 3×3 layer_blocks only (uniform width; stable). full: whole detector — often many hours.",
    )
    parser.add_argument(
        "--fpn_uniform_channels",
        type=int,
        default=None,
        help="If FPN levels disagree after prune, project all maps to this many channels (default: auto from widest sane level).",
    )

    parser.add_argument("--example_height", type=int, default=480)
    parser.add_argument("--example_width", type=int, default=640)
    parser.add_argument(
        "--graph_heartbeat_sec",
        type=float,
        default=30.0,
        help="Log every N seconds while building the dependency graph (0 = disable). Proves the process is alive.",
    )

    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--eval_conf_threshold", type=float, default=0.5)
    parser.add_argument("--eval_iou_threshold", type=float, default=0.25)
    parser.add_argument("--eval_top_k", type=int, default=1)

    parser.add_argument("--augmented_cornell_path", type=str, default="./augmented_dataset")
    parser.add_argument("--cornell_path", type=str, default="./cornell_dataset")
    parser.add_argument("--jacquard_path", type=str, default="./jacquard_dataset")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark_compile", action="store_true")
    parser.add_argument("--benchmark_runs", type=int, default=30)
    parser.add_argument("--benchmark_warmup", type=int, default=10)

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}. Prune scope: {args.prune_scope}.")
    logger.info("Loading model…")
    model = get_model(num_classes=2, freeze_backbone=args.freeze_backbone)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)

    total0, train0 = count_params(model)
    logger.info(f"Parameters before prune: total={total0:,} trainable={train0:,}")
    print_model_summary(model)

    n_fbn = replace_frozen_batchnorm_with_batchnorm2d(model)
    logger.info(f"Replaced FrozenBatchNorm2d modules: {n_fbn}")

    ex = [torch.randn(3, args.example_height, args.example_width, device=device)]
    model.eval()
    with torch.no_grad():
        _ = model(ex)
    logger.info("Sanity forward OK before pruning.")

    if args.benchmark:
        t = benchmark_forward(
            model, device, args.example_height, args.example_width,
            args.benchmark_warmup, args.benchmark_runs, use_fp16=False, use_compile=False,
        )
        logger.info(f"[Before prune] FP32 ~{t * 1000:.2f} ms / image")
        if device.type == "cuda":
            t16 = benchmark_forward(
                model, device, args.example_height, args.example_width,
                args.benchmark_warmup, args.benchmark_runs, use_fp16=True, use_compile=False,
            )
            logger.info(f"[Before prune] FP16 ~{t16 * 1000:.2f} ms / image")

    logger.info("Starting structured pruning…")
    ex_nchw = torch.randn(1, 3, args.example_height, args.example_width, device=device)
    backbone_out_after: Optional[int] = None
    structured_pruning_meta: Optional[dict] = None
    if args.prune_scope == "full":
        run_structured_prune(
            model,
            device,
            ex,
            pruning_ratio=args.pruning_ratio,
            prune_iterations=max(1, args.prune_iterations),
            round_to=args.round_to if args.round_to > 0 else None,
            isomorphic=not args.no_isomorphic,
            graph_heartbeat_sec=max(0.0, args.graph_heartbeat_sec),
        )
    else:
        run_structured_prune_fpn_only(
            model,
            device,
            ex_nchw,
            pruning_ratio=args.pruning_ratio,
            prune_iterations=max(1, args.prune_iterations),
            round_to=args.round_to if args.round_to > 0 else None,
            isomorphic=not args.no_isomorphic,
            graph_heartbeat_sec=max(0.0, args.graph_heartbeat_sec),
        )
        backbone_out_after, structured_pruning_meta = ensure_uniform_fpn_backbone(
            model,
            device,
            args.example_height,
            args.example_width,
            args.fpn_uniform_channels,
        )
        logger.info(
            f"FPN output width after prune (uniform): {backbone_out_after}. "
            "Rebuilding RPN + ROI heads to match (weights randomly initialized like a new detector head)."
        )
        rebuild_detector_heads_for_backbone_width(model, num_classes=2, backbone_out_ch=backbone_out_after)
        if args.finetune_epochs <= 0:
            logger.warning(
                "With --prune_scope backbone (FPN-only), RPN/ROI are new random weights until you fine-tune. "
                "Use e.g. --finetune_epochs 10 --lr 1e-4 (and valid dataset paths)."
            )

    total1, train1 = count_params(model)
    logger.info(f"Parameters after prune: total={total1:,} trainable={train1:,} (delta {total1 - total0:+,})")
    print_pruning_parameter_report(total0, total1)

    with torch.no_grad():
        out = model(ex)
    logger.info(f"Sanity forward after prune: {len(out)} detections, first boxes shape {out[0]['boxes'].shape}")

    if args.benchmark:
        t = benchmark_forward(
            model, device, args.example_height, args.example_width,
            args.benchmark_warmup, args.benchmark_runs, use_fp16=False, use_compile=False,
        )
        logger.info(f"[After prune] FP32 ~{t * 1000:.2f} ms / image")
        if device.type == "cuda":
            t16 = benchmark_forward(
                model, device, args.example_height, args.example_width,
                args.benchmark_warmup, args.benchmark_runs, use_fp16=True, use_compile=False,
            )
            logger.info(f"[After prune] FP16 ~{t16 * 1000:.2f} ms / image")
        if args.benchmark_compile and device.type == "cuda":
            tc = benchmark_forward(
                model.float(), device, args.example_height, args.example_width,
                args.benchmark_warmup, max(5, args.benchmark_runs // 2),
                use_fp16=False, use_compile=True,
            )
            logger.info(f"[After prune] FP32 + torch.compile ~{tc * 1000:.2f} ms / image (indicative)")

    if args.finetune_epochs > 0:
        data_sources = []
        if os.path.exists(args.augmented_cornell_path):
            data_sources.append({"type": "augmented_cornell", "path": args.augmented_cornell_path})
        if os.path.exists(args.cornell_path):
            data_sources.append({"type": "original_cornell", "path": args.cornell_path})
        if os.path.exists(args.jacquard_path):
            data_sources.append({"type": "original_jacquard", "path": args.jacquard_path})
        if not data_sources:
            logger.error("No dataset paths for fine-tuning; skipping.")
        else:
            dataset = GraspDataset(data_sources, target_size=(640, 480))
            n_train = int(args.train_ratio * len(dataset))
            train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train], generator=torch.Generator().manual_seed(42))
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

            params = [p for p in model.parameters() if p.requires_grad]
            if args.optimizer == "sgd":
                opt = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            elif args.optimizer == "adam":
                opt = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            else:
                opt = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.finetune_epochs), eta_min=args.lr * 0.01)

            for epoch in range(args.finetune_epochs):
                train_one_epoch(model, opt, train_loader, device, epoch + 1, grad_clip=0.0)
                vm = evaluate_with_diagnostics(
                    model, val_loader, device,
                    confidence_threshold=args.eval_conf_threshold,
                    iou_threshold=args.eval_iou_threshold,
                    top_k=max(1, args.eval_top_k),
                )
                sched.step()
                logger.info(f"Finetune ep {epoch+1}: val_loss={vm['avg_loss']:.4f} val_acc={vm['accuracy']:.4f}")

    total2, train2 = count_params(model)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "checkpoint_source": str(args.checkpoint),
            "prune_scope": args.prune_scope,
            "backbone_out_channels_after_prune": backbone_out_after,
            "structured_pruning_meta": structured_pruning_meta,
            "pruning_ratio": args.pruning_ratio,
            "prune_iterations": args.prune_iterations,
            "replaced_frozen_bn": n_fbn,
            "param_total_before": total0,
            "param_total_after_prune": total1,
            "param_total_final": total2,
        },
        output_dir / "structured_pruned.pth",
    )
    logger.info(f"Saved: {output_dir / 'structured_pruned.pth'}")


if __name__ == "__main__":
    main()
