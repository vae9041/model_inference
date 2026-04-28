"""
Cornell Dataset Evaluation Script for Trained Faster R-CNN Grasp Detection Model
Evaluates the model performance on Cornell grasp detection dataset
"""

import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import logging
from collections import defaultdict

# Import model from training script
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CornellEvaluator:
    """Evaluator for Cornell grasp detection dataset"""
    
    def __init__(self, model_path: str, cornell_path: str, device: str = 'auto'):
        """
        Args:
            model_path: Path to the trained model file
            cornell_path: Path to Cornell dataset
            device: Device to use for evaluation (cuda/cpu/auto)
        """
        self.cornell_path = Path(cornell_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load Cornell dataset samples
        self.samples = self._load_cornell_samples()
        
        logger.info(f"Loaded model on device: {self.device}")
        logger.info(f"Found {len(self.samples)} Cornell samples for evaluation")
    
    def _load_model(self, model_path: str):
        """Load the trained Faster R-CNN model"""
        # Create model architecture (same as in training)
        model = fasterrcnn_resnet50_fpn(weights=None)  # Updated to use weights instead of pretrained
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # Background + grasp
        
        # Load trained weights with weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def _load_cornell_samples(self) -> List[Dict]:
        """Load Cornell dataset samples"""
        samples = []
        
        for folder in sorted(self.cornell_path.iterdir()):
            if folder.is_dir() and folder.name != 'backgrounds':
                for file in folder.glob('pcd*.txt'):
                    if 'cpos' not in file.name and 'cneg' not in file.name:
                        base_name = file.stem
                        rgb_file = folder / f"{base_name}r.png"
                        depth_file = folder / f"{base_name}d.png"
                        pos_file = folder / f"{base_name}cpos.txt"
                        neg_file = folder / f"{base_name}cneg.txt"
                        
                        if all(f.exists() for f in [rgb_file, depth_file, pos_file]):
                            samples.append({
                                'rgb_path': str(rgb_file),
                                'depth_path': str(depth_file),
                                'pos_grasps_path': str(pos_file),
                                'neg_grasps_path': str(neg_file) if neg_file.exists() else None,
                                'name': base_name
                            })
        
        return samples
    
    def _load_ground_truth_grasps(self, pos_path: str, neg_path: str = None) -> List[np.ndarray]:
        """Load ground truth grasp rectangles from Cornell format"""
        grasps = []
        
        # Load positive grasps
        if os.path.exists(pos_path):
            with open(pos_path, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 4):
                    if i + 3 < len(lines):
                        corners = []
                        for j in range(4):
                            x, y = map(float, lines[i + j].strip().split())
                            corners.append([x, y])
                        grasps.append(np.array(corners))
        
        return grasps
    
    def _preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (640, 480)) -> Tuple[torch.Tensor, Tuple[float, float]]:
        """Preprocess image for model input"""
        # Load and resize image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get scaling factors
        orig_h, orig_w = image.shape[:2]
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h
        
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image_tensor, (scale_x, scale_y)
    
    def _predict_grasps(self, image_tensor: torch.Tensor, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Predict grasps for a single image"""
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            predictions = self.model(image_tensor)
        
        # Extract predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Filter by confidence
        keep = scores >= confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        
        return boxes, scores
    
    def _calculate_grasp_iou(self, pred_box: np.ndarray, gt_corners: np.ndarray) -> float:
        """Calculate IoU between predicted bounding box and ground truth grasp rectangle"""
        # Convert ground truth corners to bounding box
        gt_box = np.array([
            np.min(gt_corners[:, 0]),  # x_min
            np.min(gt_corners[:, 1]),  # y_min
            np.max(gt_corners[:, 0]),  # x_max
            np.max(gt_corners[:, 1])   # y_max
        ])
        
        # Calculate intersection
        x1 = max(pred_box[0], gt_box[0])
        y1 = max(pred_box[1], gt_box[1])
        x2 = min(pred_box[2], gt_box[2])
        y2 = min(pred_box[3], gt_box[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        union = pred_area + gt_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_metrics(self, all_predictions: List[Dict], iou_threshold: float = 0.25) -> Dict:
        """Calculate evaluation metrics"""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        total_gt_grasps = 0
        total_pred_grasps = 0
        
        ious = []
        
        for pred_data in all_predictions:
            gt_grasps = pred_data['gt_grasps']
            pred_boxes = pred_data['pred_boxes']
            pred_scores = pred_data['pred_scores']
            
            total_gt_grasps += len(gt_grasps)
            total_pred_grasps += len(pred_boxes)
            
            # Match predictions to ground truth
            gt_matched = [False] * len(gt_grasps)
            
            for pred_box, pred_score in zip(pred_boxes, pred_scores):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_grasp in enumerate(gt_grasps):
                    if gt_matched[gt_idx]:
                        continue
                    
                    iou = self._calculate_grasp_iou(pred_box, gt_grasp)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                ious.append(best_iou)
                
                if best_iou >= iou_threshold:
                    true_positives += 1
                    if best_gt_idx >= 0:
                        gt_matched[best_gt_idx] = True
                else:
                    false_positives += 1
            
            # Count unmatched ground truth as false negatives
            false_negatives += sum(1 for matched in gt_matched if not matched)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_gt_grasps': total_gt_grasps,
            'total_pred_grasps': total_pred_grasps,
            'mean_iou': np.mean(ious) if ious else 0,
            'ious': ious
        }
    
    def evaluate(self, confidence_threshold: float = 0.5, iou_threshold: float = 0.25, 
                max_samples: Optional[int] = None) -> Dict:
        """Evaluate model on Cornell dataset"""
        
        all_predictions = []
        
        samples_to_evaluate = self.samples[:max_samples] if max_samples else self.samples
        
        logger.info(f"Evaluating on {len(samples_to_evaluate)} samples...")
        
        for sample in tqdm(samples_to_evaluate, desc="Evaluating"):
            try:
                # Load and preprocess image
                image_tensor, (scale_x, scale_y) = self._preprocess_image(sample['rgb_path'])
                
                # Get predictions
                pred_boxes, pred_scores = self._predict_grasps(image_tensor, confidence_threshold)
                
                # Load ground truth
                gt_grasps = self._load_ground_truth_grasps(
                    sample['pos_grasps_path'], 
                    sample.get('neg_grasps_path')
                )
                
                # Scale ground truth to match resized image
                scaled_gt_grasps = []
                for gt_grasp in gt_grasps:
                    scaled_grasp = gt_grasp.copy()
                    scaled_grasp[:, 0] *= scale_x
                    scaled_grasp[:, 1] *= scale_y
                    scaled_gt_grasps.append(scaled_grasp)
                
                all_predictions.append({
                    'sample_name': sample['name'],
                    'pred_boxes': pred_boxes,
                    'pred_scores': pred_scores,
                    'gt_grasps': scaled_gt_grasps,
                    'image_path': sample['rgb_path'],
                    'scale_factors': (scale_x, scale_y)
                })
                
            except Exception as e:
                logger.warning(f"Error processing sample {sample['name']}: {e}")
                continue
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, iou_threshold)
        
        # Add evaluation parameters to results
        metrics['confidence_threshold'] = confidence_threshold
        metrics['iou_threshold'] = iou_threshold
        metrics['num_samples'] = len(all_predictions)
        
        return metrics, all_predictions
    
    def evaluate_success_rate(self, confidence_threshold: float = 0.5, iou_threshold: float = 0.25,
                             max_samples: Optional[int] = None) -> Dict:
        """
        Evaluate using IoU-based success rate.
        For each image, select the best predicted grasp and check if it has IoU >= threshold with any GT grasp.
        Returns: Accuracy % = Correct predictions / Total images
        """
        samples_to_evaluate = self.samples[:max_samples] if max_samples else self.samples
        
        correct_predictions = 0
        total_images = 0
        detailed_results = []
        
        logger.info(f"Evaluating IoU-based success rate on {len(samples_to_evaluate)} samples...")
        
        for sample in tqdm(samples_to_evaluate, desc="Evaluating Success Rate"):
            try:
                total_images += 1
                
                # Load and preprocess image
                image_tensor, (scale_x, scale_y) = self._preprocess_image(sample['rgb_path'])
                
                # Get predictions
                pred_boxes, pred_scores = self._predict_grasps(image_tensor, confidence_threshold)
                
                # Skip if no confident predictions
                if len(pred_boxes) == 0:
                    detailed_results.append({
                        'sample_name': sample['name'],
                        'success': False,
                        'reason': 'No confident predictions',
                        'best_iou': 0.0,
                        'num_predictions': 0,
                        'num_gt_grasps': 0
                    })
                    continue
                
                # Select best prediction (highest confidence)
                best_pred_idx = np.argmax(pred_scores)
                best_pred_box = pred_boxes[best_pred_idx]
                best_pred_score = pred_scores[best_pred_idx]
                
                # Convert best prediction to grasp rectangle format
                from train import GraspRectangle  # Import from training script
                best_pred_grasp = GraspRectangle.from_bbox(
                    best_pred_box[0], best_pred_box[1],
                    best_pred_box[2], best_pred_box[3]
                )
                
                # Load ground truth grasps
                gt_grasps = self._load_ground_truth_grasps(
                    sample['pos_grasps_path'], 
                    sample.get('neg_grasps_path')
                )
                
                # Check IoU with all ground truth grasps
                max_iou = 0.0
                for gt_grasp_corners in gt_grasps:
                    # Scale ground truth to match resized image
                    scaled_gt_corners = gt_grasp_corners.copy()
                    scaled_gt_corners[:, 0] *= scale_x
                    scaled_gt_corners[:, 1] *= scale_y
                    
                    # Convert GT corners to bounding box for IoU calculation
                    x_coords = scaled_gt_corners[:, 0]
                    y_coords = scaled_gt_corners[:, 1]
                    gt_bbox = [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]
                    
                    gt_grasp = GraspRectangle.from_bbox(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3])
                    iou = best_pred_grasp.intersection_over_union(gt_grasp)
                    max_iou = max(max_iou, iou)
                
                # Count as correct if IoU >= threshold
                success = max_iou >= iou_threshold
                if success:
                    correct_predictions += 1
                
                detailed_results.append({
                    'sample_name': sample['name'],
                    'success': success,
                    'best_iou': max_iou,
                    'best_score': best_pred_score,
                    'num_predictions': len(pred_boxes),
                    'num_gt_grasps': len(gt_grasps),
                    'reason': 'Success' if success else f'IoU {max_iou:.3f} < {iou_threshold}'
                })
                
            except Exception as e:
                logger.warning(f"Error processing sample {sample['name']}: {e}")
                detailed_results.append({
                    'sample_name': sample['name'],
                    'success': False,
                    'reason': f'Error: {str(e)}',
                    'best_iou': 0.0,
                    'num_predictions': 0,
                    'num_gt_grasps': 0
                })
                continue
        
        # Calculate final metrics
        accuracy = correct_predictions / total_images if total_images > 0 else 0.0
        
        results = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_images': total_images,
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
            'detailed_results': detailed_results
        }
        
        return results
    
    def visualize_predictions(self, predictions: List[Dict], output_dir: str, num_samples: int = 10):
        """Visualize predictions on sample images"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, pred_data in enumerate(predictions[:num_samples]):
            # Load original image
            image = cv2.imread(pred_data['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to match prediction scale
            image = cv2.resize(image, (640, 480))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Ground truth visualization
            ax1.imshow(image)
            ax1.set_title('Ground Truth Grasps')
            for gt_grasp in pred_data['gt_grasps']:
                # Draw grasp rectangle
                polygon = patches.Polygon(gt_grasp, linewidth=2, edgecolor='green', facecolor='none')
                ax1.add_patch(polygon)
            ax1.axis('off')
            
            # Predictions visualization
            ax2.imshow(image)
            ax2.set_title('Predicted Grasps')
            for pred_box, score in zip(pred_data['pred_boxes'], pred_data['pred_scores']):
                # Draw bounding box
                rect = patches.Rectangle(
                    (pred_box[0], pred_box[1]), 
                    pred_box[2] - pred_box[0], 
                    pred_box[3] - pred_box[1],
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax2.add_patch(rect)
                # Add confidence score
                ax2.text(pred_box[0], pred_box[1] - 5, f'{score:.2f}', 
                        fontsize=10, color='red', weight='bold')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path / f'{pred_data["sample_name"]}_predictions.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Saved {min(num_samples, len(predictions))} visualization images to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Faster R-CNN on Cornell Grasp Dataset')
    parser.add_argument('--model_path', type=str, default='./trained_models/best_model.pth',
                       help='Path to trained model file')
    parser.add_argument('--cornell_path', type=str, default='./cornell_dataset',
                       help='Path to Cornell dataset')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--iou_threshold', type=float, default=0.25,
                       help='IoU threshold for positive matches')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None for all)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization images')
    parser.add_argument('--num_visualizations', type=int, default=10,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = CornellEvaluator(args.model_path, args.cornell_path, args.device)
    
    # Run detailed evaluation (per-grasp metrics)
    logger.info("Starting detailed evaluation...")
    metrics, predictions = evaluator.evaluate(
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        max_samples=args.max_samples
    )
    
    # Run IoU-based success rate evaluation (per-image metrics)
    logger.info("Starting IoU-based success rate evaluation...")
    success_metrics = evaluator.evaluate_success_rate(
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        max_samples=args.max_samples
    )
    
    # Print results
    print("\n" + "="*60)
    print("CORNELL DATASET EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.cornell_path}")
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"Confidence threshold: {metrics['confidence_threshold']}")
    print(f"IoU threshold: {metrics['iou_threshold']}")
    
    print("\n" + "="*20 + " PER-GRASP METRICS " + "="*20)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print("-"*60)
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"Total GT Grasps: {metrics['total_gt_grasps']}")
    print(f"Total Predictions: {metrics['total_pred_grasps']}")
    
    print("\n" + "="*18 + " IoU-BASED SUCCESS RATE " + "="*18)
    print(f"Success Rate (Accuracy): {success_metrics['accuracy']:.4f} ({success_metrics['accuracy']*100:.2f}%)")
    print(f"Correct Predictions: {success_metrics['correct_predictions']}")
    print(f"Total Images: {success_metrics['total_images']}")
    print("="*60)
    
    # Save detailed results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        # Convert all metrics to JSON-serializable format
        def make_json_serializable(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        # Remove numpy arrays and convert types for JSON serialization
        json_metrics = {k: make_json_serializable(v) for k, v in metrics.items() if k != 'ious'}
        json_success_metrics = make_json_serializable(success_metrics)
                
        combined_results = {
            'per_grasp_metrics': json_metrics,
            'success_rate_metrics': json_success_metrics
        }
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Generate visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        evaluator.visualize_predictions(
            predictions, 
            output_dir / 'visualizations',
            args.num_visualizations
        )
    
    # Plot IoU distribution
    if metrics['ious']:
        plt.figure(figsize=(10, 6))
        plt.hist(metrics['ious'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(args.iou_threshold, color='red', linestyle='--', 
                   label=f'IoU Threshold ({args.iou_threshold})')
        plt.xlabel('IoU Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of IoU Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'iou_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Evaluation completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
