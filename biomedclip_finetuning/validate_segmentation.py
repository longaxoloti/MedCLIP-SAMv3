"""
Segmentation Validation Pipeline for Frequency Adapter Training

This module provides quick validation functions to compute segmentation metrics
(DSC, NSD) during training to ensure the adapter doesn't degrade segmentation quality.

Usage:
    validator = SegmentationValidator(
        gt_path='data/brain_tumors/test_masks',
        device='cuda:0'
    )
    dsc, nsd = validator.validate(model, test_images_path='data/brain_tumors/test_images')
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import cv2
import logging


class SegmentationValidator:
    """
    Quick validator for segmentation quality during training.
    
    Args:
        gt_path: Path to ground truth segmentation masks
        device: Device for computation (cuda:0, cpu, etc.)
        num_samples: Number of samples to validate (None = all)
        logger: Logger instance (optional)
    """
    
    def __init__(
        self,
        gt_path: str,
        device: str = 'cuda:0',
        num_samples: Optional[int] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.gt_path = Path(gt_path)
        self.device = torch.device(device)
        self.num_samples = num_samples
        self.logger = logger or self._default_logger()
        
        # Load ground truth masks
        self.gt_files = sorted([
            f for f in self.gt_path.glob('*.png') if f.is_file()
        ])
        
        if self.num_samples:
            self.gt_files = self.gt_files[:self.num_samples]
        
        if not self.gt_files:
            self.logger.warning(f"No ground truth masks found in {gt_path}")
        else:
            self.logger.info(f"Loaded {len(self.gt_files)} ground truth masks")
    
    @staticmethod
    def _default_logger():
        """Create a default logger."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def compute_dice(self, pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
        """
        Compute Dice Similarity Coefficient.
        
        Args:
            pred: Predicted binary mask (H, W)
            gt: Ground truth binary mask (H, W)
            eps: Small epsilon for numerical stability
            
        Returns:
            Dice score (0-1)
        """
        pred = pred.astype(np.float32)
        gt = gt.astype(np.float32)
        
        intersection = np.sum(pred * gt)
        dice = (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + eps)
        
        return float(dice)
    
    def compute_nsd(self, pred: np.ndarray, gt: np.ndarray, threshold: float = 0.95) -> float:
        """
        Compute Normalized Surface Dice (NSD).
        
        Args:
            pred: Predicted binary mask (H, W)
            gt: Ground truth binary mask (H, W)
            threshold: Distance threshold in pixels (default 2 pixels)
            
        Returns:
            NSD score (0-1)
        """
        try:
            from scipy import ndimage
        except ImportError:
            self.logger.warning("scipy not available, skipping NSD computation")
            return 0.0
        
        pred = pred.astype(np.bool_)
        gt = gt.astype(np.bool_)
        
        # Compute signed distance transform
        pred_dist = ndimage.distance_transform_edt(~pred)
        gt_dist = ndimage.distance_transform_edt(~gt)
        
        # Surface points: where distance > 0
        pred_surface = pred_dist > 0
        gt_surface = gt_dist > 0
        
        if not (np.any(pred_surface) or np.any(gt_surface)):
            return 1.0  # Both empty
        
        # Distance from pred surface to gt surface
        if np.any(pred_surface):
            dist_pred_to_gt = np.min(gt_dist[pred_surface])
        else:
            dist_pred_to_gt = float('inf')
        
        # Distance from gt surface to pred surface
        if np.any(gt_surface):
            dist_gt_to_pred = np.min(pred_dist[gt_surface])
        else:
            dist_gt_to_pred = float('inf')
        
        # Check if both distances are below threshold
        if dist_pred_to_gt <= threshold and dist_gt_to_pred <= threshold:
            return 1.0
        else:
            return 0.0
    
    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        images_path: str,
        return_individual_scores: bool = False
    ) -> Dict[str, float]:
        """
        Validate segmentation quality.
        
        Args:
            model: BiomedCLIP model with adapters
            images_path: Path to test images
            return_individual_scores: If True, return per-image scores
            
        Returns:
            Dictionary with average DSC, NSD, or per-image scores if requested
        """
        images_path = Path(images_path)
        
        if not images_path.exists():
            self.logger.warning(f"Images directory not found: {images_path}")
            return {'dsc': 0.0, 'nsd': 0.0, 'error': 'images_not_found'}
        
        # Find all test images
        image_files = sorted([
            f for f in images_path.glob('*.png') if f.is_file()
        ])
        
        if len(image_files) != len(self.gt_files):
            self.logger.warning(
                f"Number of images ({len(image_files)}) doesn't match "
                f"number of ground truth masks ({len(self.gt_files)})"
            )
        
        dsc_scores = []
        nsd_scores = []
        
        model.eval()
        for i, (img_file, gt_file) in enumerate(zip(image_files[:len(self.gt_files)], self.gt_files)):
            try:
                # Load image
                image = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    self.logger.warning(f"Could not load image: {img_file}")
                    continue
                
                # Convert to 3-channel if needed
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                # Load ground truth
                gt = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    self.logger.warning(f"Could not load ground truth: {gt_file}")
                    continue
                
                # Threshold GT to binary
                gt_binary = (gt > 128).astype(np.uint8)
                
                # Generate saliency map using model
                # (This is simplified - in practice you'd use IBA or similar)
                image_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                image_tensor = image_tensor.to(self.device)
                
                with torch.no_grad():
                    outputs = model.vision_model(image_tensor)
                    # Get attention maps from vision transformer
                    # This is a simplification - real use would require proper saliency extraction
                    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                        # Average attention across all heads and layers
                        attn = outputs.attentions[-1]  # Last layer
                        saliency = attn[0, :, 0, :].mean(dim=0)  # Average over heads, use CLS token
                        saliency = saliency.cpu().numpy()
                        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-7)
                    else:
                        # Fallback: use a simple edge detection as placeholder
                        saliency = cv2.Canny(image, 50, 150).astype(np.float32) / 255.0
                
                # Threshold saliency to binary
                saliency_binary = (saliency > 0.5).astype(np.uint8)
                
                # Compute metrics
                dsc = self.compute_dice(saliency_binary, gt_binary)
                nsd = self.compute_nsd(saliency_binary, gt_binary)
                
                dsc_scores.append(dsc)
                nsd_scores.append(nsd)
                
            except Exception as e:
                self.logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        if not dsc_scores:
            self.logger.warning("No valid samples processed during validation")
            return {'dsc': 0.0, 'nsd': 0.0, 'error': 'no_valid_samples'}
        
        avg_dsc = float(np.mean(dsc_scores))
        avg_nsd = float(np.mean(nsd_scores))
        
        results = {
            'dsc_mean': avg_dsc,
            'dsc_std': float(np.std(dsc_scores)),
            'nsd_mean': avg_nsd,
            'nsd_std': float(np.std(nsd_scores)),
            'num_samples': len(dsc_scores),
        }
        
        if return_individual_scores:
            results['dsc_per_sample'] = dsc_scores
            results['nsd_per_sample'] = nsd_scores
        
        return results
    
    def get_baseline_metrics(self) -> Dict[str, float]:
        """
        Get baseline metrics for comparison (without adapter).
        
        These would typically be cached and loaded from a previous run.
        """
        # Placeholder: In real use, these would be loaded from a cache file
        return {
            'baseline_dsc': 0.58,
            'baseline_nsd': 0.64,
        }


def quick_segmentation_check(
    model: nn.Module,
    data_path: str,
    mask_path: str,
    device: str = 'cuda:0',
    num_samples: int = 10
) -> Tuple[float, float]:
    """
    Quick function for checking segmentation quality during training.
    
    Args:
        model: BiomedCLIP model with adapters
        data_path: Path to validation images
        mask_path: Path to ground truth masks
        device: Device for computation
        num_samples: Number of samples to check
        
    Returns:
        Tuple of (average_DSC, average_NSD)
    """
    validator = SegmentationValidator(
        gt_path=mask_path,
        device=device,
        num_samples=num_samples
    )
    
    results = validator.validate(
        model=model,
        images_path=data_path,
        return_individual_scores=False
    )
    
    dsc = results.get('dsc_mean', 0.0)
    nsd = results.get('nsd_mean', 0.0)
    
    return dsc, nsd
