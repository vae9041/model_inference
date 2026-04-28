#!/usr/bin/env python3
"""
Jacquard Dataset Evaluation Script for Faster R-CNN Grasp Detection (ResNet-18 backbone).

This script reuses the existing `eval_jacquard.py` evaluation pipeline and CLI,
but swaps model construction to the ResNet-18 Faster R-CNN architecture from
`train_resnet_18.py`.
"""

import eval_jacquard as base_eval
from train_resnet_18 import get_model as get_resnet18_model


def get_model(num_classes: int = 2):
    """Create Faster R-CNN model with ResNet-18 FPN backbone."""
    return get_resnet18_model(num_classes=num_classes, freeze_backbone=False)


# Override the model builder used by eval_jacquard's evaluator.
base_eval.get_model = get_model


if __name__ == "__main__":
    base_eval.main()

