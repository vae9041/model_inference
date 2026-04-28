"""
Faster R-CNN Training Script (ResNet-18 FPN backbone) for Grasp Detection

This script mirrors `train.py` but uses a ResNet-18 + FPN feature extractor
instead of the ResNet-50 one.
"""

import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

import matplotlib.pyplot as plt

from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet18_Weights

# Reuse your existing data + training utilities
from train import (
    GraspDataset,
    GraspRectangle,
    collate_fn,
    train_one_epoch,
    freeze_backbone_parameters,
    print_model_info,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model(num_classes: int, freeze_backbone: bool = False) -> torch.nn.Module:
    """
    Create Faster R-CNN model with ResNet-18 FPN backbone.

    Notes:
    - The Faster R-CNN detection head is initialized randomly.
    - The ResNet-18 backbone uses ImageNet pretrained weights.
    """
    # Build the backbone with FPN on top of a pretrained ResNet-18.
    # `trainable_layers` controls how many late ResNet stages remain trainable
    # inside `resnet_fpn_backbone` before the optional `freeze_backbone` flag.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet18",
        weights=ResNet18_Weights.IMAGENET1K_V1,
        trainable_layers=3,
    )

    model = FasterRCNN(backbone, num_classes=num_classes)

    if freeze_backbone:
        # Freeze the full backbone feature extractor (ResNet + FPN),
        # leaving RPN + ROI head trainable.
        freeze_backbone_parameters(model)

    return model


def initialize_detector_from_resnet50(model: torch.nn.Module) -> int:
    """
    Initialize RPN / ROI-head compatible weights from a stronger COCO detector.

    Returns:
        Number of parameter tensors copied into the target model.
    """
    try:
        baseline = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    except Exception:
        # Fallback for older torchvision releases.
        baseline = fasterrcnn_resnet50_fpn(pretrained=True)

    src_state = baseline.state_dict()
    dst_state = model.state_dict()
    copied = 0

    for name, src_tensor in src_state.items():
        # Keep custom backbone untouched; transfer detector-specific modules only.
        if not (name.startswith("rpn.") or name.startswith("roi_heads.")):
            continue
        if name in dst_state and dst_state[name].shape == src_tensor.shape:
            dst_state[name] = src_tensor
            copied += 1

    model.load_state_dict(dst_state)
    return copied


def evaluate_with_diagnostics(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.25,
    top_k: int = 1,
):
    """
    Evaluate model using IoU-based success rate with diagnostics.

    Success rule is preserved:
      A prediction is counted as correct only if IoU with any GT >= iou_threshold.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_images = 0
    correct_predictions = 0
    images_with_any_pred = 0
    images_with_pred_at_conf = 0
    max_scores = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute validation loss
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1

            # Compute predictions
            model.eval()
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                total_images += 1
                scores = pred["scores"]
                boxes = pred["boxes"]

                if scores.numel() > 0:
                    images_with_any_pred += 1
                    max_scores.append(float(scores.max().item()))
                else:
                    max_scores.append(0.0)

                keep = scores >= confidence_threshold
                kept_boxes = boxes[keep]
                kept_scores = scores[keep]

                if kept_boxes.shape[0] == 0:
                    continue
                images_with_pred_at_conf += 1

                k = min(top_k, kept_boxes.shape[0])
                topk_indices = torch.topk(kept_scores, k=k).indices
                candidate_boxes = kept_boxes[topk_indices].cpu().numpy()
                gt_boxes = target["boxes"].cpu().numpy()

                image_success = False
                for box in candidate_boxes:
                    pred_grasp = GraspRectangle.from_bbox(box[0], box[1], box[2], box[3])
                    max_iou = 0.0
                    for gt_box in gt_boxes:
                        gt_grasp = GraspRectangle.from_bbox(gt_box[0], gt_box[1], gt_box[2], gt_box[3])
                        iou = pred_grasp.intersection_over_union(gt_grasp)
                        max_iou = max(max_iou, iou)
                    # Keep the exact IoU success requirement.
                    if max_iou >= iou_threshold:
                        image_success = True
                        break

                if image_success:
                    correct_predictions += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct_predictions / total_images if total_images > 0 else 0.0
    score_array = np.array(max_scores, dtype=np.float32) if max_scores else np.array([0.0], dtype=np.float32)

    return {
        "avg_loss": avg_loss,
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_images": total_images,
        "images_with_any_pred": images_with_any_pred,
        "images_with_pred_at_conf": images_with_pred_at_conf,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "top_k": top_k,
        "score_mean": float(score_array.mean()),
        "score_median": float(np.median(score_array)),
        "score_p90": float(np.percentile(score_array, 90)),
        "score_max": float(score_array.max()),
    }


def print_model_summary(model: torch.nn.Module) -> None:
    """Print a concise model summary (ResNet-18 version)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("Model: Faster R-CNN with ResNet-18 FPN backbone")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: ~{total_params * 4 / (1024 * 1024):.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for Grasp Detection (ResNet-18)")
    parser.add_argument("--augmented_cornell_path", type=str, default="./augmented_dataset",
                        help="Path to augmented Cornell dataset")
    parser.add_argument("--cornell_path", type=str, default="./cornell_dataset",
                        help="Path to original Cornell dataset")
    parser.add_argument("--jacquard_path", type=str, default="./jacquard_dataset",
                        help="Path to original Jacquard dataset")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0003, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")

    # Learning rate scheduler options
    parser.add_argument(
        "--scheduler",
        type=str,
        default="step",
        choices=["step", "cosine", "plateau", "exponential"],
        help="Learning rate scheduler type",
    )
    parser.add_argument("--step_size", type=int, default=20, help="Step size for StepLR scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR and ExponentialLR schedulers")

    # Optimizer options
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"],
                        help="Optimizer type")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")

    # Training improvements
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of linear warmup epochs")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1,
                        help="Warmup starts at lr * warmup_start_factor")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping threshold (0 = no clipping)")
    parser.add_argument("--init_from_resnet50_detector_head", action="store_true",
                        help="Initialize RPN/ROI head weights from pretrained ResNet-50 Faster R-CNN")

    # Evaluation controls
    parser.add_argument("--eval_conf_threshold", type=float, default=0.5,
                        help="Confidence threshold used during validation")
    parser.add_argument("--eval_iou_threshold", type=float, default=0.25,
                        help="IoU threshold for success criterion (keep at 0.25 for your metric)")
    parser.add_argument("--eval_top_k", type=int, default=1,
                        help="Evaluate success over top-k predictions above threshold")

    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save trained models")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data to use for training")
    parser.add_argument("--show_model_details", action="store_true", help="Show detailed model architecture and parameters")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the backbone feature extractor (only train RPN and ROI head)")

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define data sources
    data_sources = []

    if os.path.exists(args.augmented_cornell_path):
        data_sources.append({"type": "augmented_cornell", "path": args.augmented_cornell_path})
    if os.path.exists(args.cornell_path):
        data_sources.append({"type": "original_cornell", "path": args.cornell_path})
    if os.path.exists(args.jacquard_path):
        data_sources.append({"type": "original_jacquard", "path": args.jacquard_path})

    if not data_sources:
        logger.error("No valid data sources found!")
        return

    logger.info(f"Using data sources: {[s['type'] for s in data_sources]}")

    # Create dataset
    dataset = GraspDataset(data_sources, target_size=(640, 480))

    # Split dataset into train and validation
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Create model
    model = get_model(num_classes=2, freeze_backbone=args.freeze_backbone)  # Background + grasp
    if args.init_from_resnet50_detector_head:
        copied_tensors = initialize_detector_from_resnet50(model)
        logger.info(f"Initialized detector heads from ResNet-50 baseline: {copied_tensors} tensors copied")
    model.to(device)

    # Print model information
    if args.show_model_details:
        print_model_info(model)
    else:
        print_model_summary(model)

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Learning rate scheduler
    effective_epochs = max(1, args.num_epochs - args.warmup_epochs)
    if args.scheduler == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=effective_epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == "plateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    elif args.scheduler == "exponential":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Training loop
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []

    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        # Optional linear warmup on base LR.
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            alpha = float(epoch + 1) / float(args.warmup_epochs)
            warmup_factor = args.warmup_start_factor + alpha * (1.0 - args.warmup_start_factor)
            current_lr = args.lr * warmup_factor
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            logger.info(f"Warmup LR set to {current_lr:.8f} (epoch {epoch + 1}/{args.warmup_epochs})")

        # Train
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch + 1,
            grad_clip=args.grad_clip,
        )
        train_loss = train_metrics["total_loss"]
        train_losses.append(train_loss)

        # Validate with IoU-based accuracy
        val_metrics = evaluate_with_diagnostics(
            model,
            val_loader,
            device,
            confidence_threshold=args.eval_conf_threshold,
            iou_threshold=args.eval_iou_threshold,
            top_k=max(1, args.eval_top_k),
        )
        val_loss = val_metrics["avg_loss"]
        val_accuracy = val_metrics["accuracy"]

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Update learning rate
        if epoch >= args.warmup_epochs:
            if args.scheduler == "plateau":
                lr_scheduler.step(val_loss)  # ReduceLROnPlateau needs the metric
            else:
                lr_scheduler.step()

        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(
            f"Val Accuracy: {val_accuracy:.4f} "
            f"({val_metrics['correct_predictions']}/{val_metrics['total_images']})"
        )
        logger.info(
            f"Eval diagnostics: top_k={val_metrics['top_k']} conf>={val_metrics['confidence_threshold']:.2f} "
            f"images_with_preds={val_metrics['images_with_pred_at_conf']}/{val_metrics['total_images']} "
            f"any_preds={val_metrics['images_with_any_pred']}/{val_metrics['total_images']}"
        )
        logger.info(
            f"Confidence distribution (max score/image): mean={val_metrics['score_mean']:.4f}, "
            f"median={val_metrics['score_median']:.4f}, p90={val_metrics['score_p90']:.4f}, max={val_metrics['score_max']:.4f}"
        )

        # Save best model based on accuracy (or loss if you prefer)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                output_dir / "best_model.pth",
            )
            logger.info(f"New best model saved with validation accuracy: {val_accuracy:.4f}")

        # Also save if loss improved but accuracy didn't (fallback)
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                output_dir / "best_loss_model.pth",
            )
            logger.info(f"Best loss model saved with validation loss: {val_loss:.4f}")

        # Save latest model
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                output_dir / f"model_epoch_{epoch + 1}.pth",
            )

    # Save final model
    torch.save(
        {
            "epoch": args.num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        },
        output_dir / "final_model.pth",
    )

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot losses
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    # Plot accuracy
    ax2.plot(val_accuracies, label="Validation Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU-based Accuracy")
    ax2.set_title("Validation IoU-based Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png")
    plt.close()

    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()

