"""
Cornell Dataset Evaluation Script for Trained Faster R-CNN (ResNet-18 backbone)
Evaluates the model performance on Cornell grasp detection dataset.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_cornell import CornellEvaluator as BaseCornellEvaluator
from structured_pruning_resnet_18 import load_structured_pruned_state_dict
from train_resnet_18 import get_model


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CornellEvaluatorRes18(BaseCornellEvaluator):
    """Cornell evaluator that loads ResNet-18 Faster R-CNN checkpoints."""

    def _load_model(self, model_path: str):
        # Create model architecture from ResNet-18 training script
        model = get_model(num_classes=2, freeze_backbone=False)

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            if isinstance(checkpoint, dict) and checkpoint.get("structured_pruning_meta"):
                load_structured_pruned_state_dict(model, checkpoint, self.device)
                logger.info("Loaded structured-pruned checkpoint with FPN wrapper metadata")
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)

        return model


def make_json_serializable(obj):
    """Convert nested numpy / torch scalar-like values for JSON serialization."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN ResNet-18 on Cornell Grasp Dataset")
    parser.add_argument("--model_path", type=str, default="./models_resnet18/best_model.pth",
                        help="Path to trained ResNet-18 model file")
    parser.add_argument("--cornell_path", type=str, default="./cornell_dataset",
                        help="Path to Cornell dataset")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Confidence threshold for predictions")
    parser.add_argument("--iou_threshold", type=float, default=0.25,
                        help="IoU threshold for positive matches")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (None for all)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results_res18",
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization images")
    parser.add_argument("--num_visualizations", type=int, default=10,
                        help="Number of samples to visualize")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = CornellEvaluatorRes18(args.model_path, args.cornell_path, args.device)

    logger.info("Starting detailed evaluation...")
    metrics, predictions = evaluator.evaluate(
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        max_samples=args.max_samples,
    )

    logger.info("Starting IoU-based success rate evaluation...")
    success_metrics = evaluator.evaluate_success_rate(
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 60)
    print("CORNELL DATASET EVALUATION RESULTS (RESNET-18)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.cornell_path}")
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"Confidence threshold: {metrics['confidence_threshold']}")
    print(f"IoU threshold: {metrics['iou_threshold']}")

    print("\n" + "=" * 20 + " PER-GRASP METRICS " + "=" * 20)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print("-" * 60)
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Total GT Grasps: {metrics['total_gt_grasps']}")
    print(f"Total Predictions: {metrics['total_pred_grasps']}")

    print("\n" + "=" * 18 + " IoU-BASED SUCCESS RATE " + "=" * 18)
    print(f"Success Rate (Accuracy): {success_metrics['accuracy']:.4f} ({success_metrics['accuracy'] * 100:.2f}%)")
    print(f"Correct Predictions: {success_metrics['correct_predictions']}")
    print(f"Total Images: {success_metrics['total_images']}")
    print("=" * 60)

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json_metrics = {k: make_json_serializable(v) for k, v in metrics.items() if k != "ious"}
        json_success_metrics = make_json_serializable(success_metrics)
        combined_results = {
            "per_grasp_metrics": json_metrics,
            "success_rate_metrics": json_success_metrics,
        }
        json.dump(combined_results, f, indent=2)

    logger.info(f"Detailed results saved to: {results_file}")

    if args.visualize:
        logger.info("Generating visualizations...")
        evaluator.visualize_predictions(
            predictions,
            output_dir / "visualizations",
            args.num_visualizations,
        )

    if metrics["ious"]:
        plt.figure(figsize=(10, 6))
        plt.hist(metrics["ious"], bins=50, alpha=0.7, edgecolor="black")
        plt.axvline(args.iou_threshold, color="red", linestyle="--", label=f"IoU Threshold ({args.iou_threshold})")
        plt.xlabel("IoU Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of IoU Scores")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "iou_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Evaluation completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

