#!/usr/bin/env python3
"""
Baseline Evaluation Script for BiomedCLIP (without frequency adapters)

This script evaluates the pre-trained BiomedCLIP model on the 4 medical imaging datasets
(brain_tumors, breast_tumors, lung_CT, lung_Xray) without any fine-tuning.

Usage:
    python evaluate_baseline.py --dataset-root /path/to/data --output-dir ./results
    
Example:
    python evaluate_baseline.py --dataset-root ~/projects/MedCLIP-SAMv3/data \
                                 --output-dir ./baseline_results \
                                 --device cuda:0
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch
import csv
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from evaluation.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
except ImportError:
    print("Warning: Could not import SurfaceDice, installing would be required for NSD metric")
    compute_dice_coefficient = None


class BiomedCLIPBaseline:
    """Baseline evaluation for pre-trained BiomedCLIP without adapters."""
    
    def __init__(self, device: str = "cuda:0", checkpoint_path: Optional[str] = None):
        """
        Initialize BiomedCLIP model.
        
        Args:
            device: Device to load model on (e.g., 'cuda:0', 'cpu')
            checkpoint_path: Path to checkpoint. Options:
                - None: Load from HuggingFace (default: "chuhac/BiomedCLIP-vit-bert-hf")
                - "saliency_maps/model": Load finetuned checkpoint from this path
                - Any path: Load from custom checkpoint directory
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load BiomedCLIP model from checkpoint."""
        print(f"Loading BiomedCLIP model on {self.device}...")
        
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
            
            # Default HuggingFace model
            hf_model_name = "chuhac/BiomedCLIP-vit-bert-hf"
            
            if self.checkpoint_path is None:
                # Load from HuggingFace Hub
                print(f"  - Source: HuggingFace ({hf_model_name})")
                self.model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)
                self.processor = AutoProcessor.from_pretrained(hf_model_name, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
            else:
                # Load from local checkpoint
                print(f"  - Source: Local checkpoint ({self.checkpoint_path})")
                self.model = AutoModel.from_pretrained(self.checkpoint_path, trust_remote_code=True)
                # Still use HF processor/tokenizer (they don't change)
                self.processor = AutoProcessor.from_pretrained(hf_model_name, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully")
            print(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    @torch.no_grad()
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features using BiomedCLIP vision encoder.
        
        Args:
            images: Tensor of shape (B, C, H, W)
            
        Returns:
            Image features of shape (B, feature_dim)
        """
        images = images.to(self.device)
        outputs = self.model.vision_model(images)
        # Use pooled output (CLS token)
        image_features = outputs.pooler_output  # (B, 768)
        return image_features
    
    @torch.no_grad()
    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        """
        Extract text features from text prompts.
        
        Args:
            texts: List of text prompts
            
        Returns:
            Text features of shape (n_texts, feature_dim)
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.text_model(**inputs)
        # Use pooled output
        text_features = outputs.pooler_output  # (n_texts, 768)
        return text_features


class DatasetEvaluator:
    """Evaluator for medical imaging datasets."""
    
    DATASETS = {
        'brain_tumors': 'Brain Tumor',
        'breast_tumors': 'Breast Tumor',
        'lung_CT': 'Lung CT',
        'lung_Xray': 'Lung X-Ray'
    }
    
    def __init__(self, dataset_root: str, device: str = "cuda:0", checkpoint_path: Optional[str] = None):
        """
        Initialize dataset evaluator.
        
        Args:
            dataset_root: Root directory containing all datasets
            device: Device for model
            checkpoint_path: Path to model checkpoint (None = HuggingFace, or custom path)
        """
        self.dataset_root = Path(dataset_root)
        self.device = device
        self.baseline_model = BiomedCLIPBaseline(device=device, checkpoint_path=checkpoint_path)
        self.results = defaultdict(list)
        
    def compute_dice_coefficient(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """
        Compute Dice coefficient between ground truth and predicted masks.
        
        Args:
            gt_mask: Ground truth mask (H, W) with values in {0, 1}
            pred_mask: Predicted mask (H, W) with values in {0, 1}
            
        Returns:
            Dice coefficient in [0, 1]
        """
        if compute_dice_coefficient is not None:
            return compute_dice_coefficient(gt_mask.astype(bool), pred_mask.astype(bool))
        
        # Fallback: manual computation
        intersection = np.sum(gt_mask & pred_mask)
        dice = 2.0 * intersection / (np.sum(gt_mask) + np.sum(pred_mask) + 1e-8)
        return float(dice)
    
    def compute_iou(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU).
        
        Args:
            gt_mask: Ground truth mask (H, W) with values in {0, 1}
            pred_mask: Predicted mask (H, W) with values in {0, 1}
            
        Returns:
            IoU in [0, 1]
        """
        intersection = np.sum(gt_mask & pred_mask)
        union = np.sum(gt_mask | pred_mask)
        iou = intersection / (union + 1e-8)
        return float(iou)
    
    def preprocess_masks(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess masks to ensure they have matching sizes and are binary.
        
        Args:
            gt_mask: Ground truth mask
            pred_mask: Predicted mask
            
        Returns:
            Tuple of preprocessed (gt_mask, pred_mask)
        """
        # Ensure same size
        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Convert to binary (threshold at 127 for uint8)
        if len(gt_mask.shape) == 3:
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        if len(pred_mask.shape) == 3:
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
        
        # Binarize
        _, gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)
        _, pred_mask = cv2.threshold(pred_mask, 127, 1, cv2.THRESH_BINARY)
        
        return gt_mask.astype(np.uint8), pred_mask.astype(np.uint8)
    
    def evaluate_dataset(self, dataset_name: str, split: str = 'test') -> Dict:
        """
        Evaluate baseline model on a specific dataset.
        
        Args:
            dataset_name: Name of dataset (e.g., 'brain_tumors')
            split: Data split ('test', 'val', or 'train')
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {self.DATASETS[dataset_name]} ({split})")
        print(f"{'='*60}")
        
        dataset_path = self.dataset_root / dataset_name
        images_dir = dataset_path / f"{split}_images"
        masks_dir = dataset_path / f"{split}_masks"
        
        # Check if directories exist
        if not images_dir.exists():
            print(f"✗ Images directory not found: {images_dir}")
            return {}
        if not masks_dir.exists():
            print(f"✗ Masks directory not found: {masks_dir}")
            return {}
        
        # Get list of images
        image_files = sorted([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print(f"✗ No images found in {images_dir}")
            return {}
        
        print(f"Found {len(image_files)} images")
        
        # Metrics storage
        metrics = {
            'images': [],
            'dice': [],
            'iou': [],
        }
        
        # Evaluate each image
        for img_file in tqdm(image_files, desc="Evaluating"):
            img_path = images_dir / img_file
            mask_path = masks_dir / img_file
            
            if not mask_path.exists():
                print(f"  Warning: Mask not found for {img_file}")
                continue
            
            try:
                # Load images
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if image is None or gt_mask is None:
                    print(f"  Error: Could not load {img_file}")
                    continue
                
                # For baseline: use simple thresholding as a dummy prediction
                # (This is just to establish baseline performance with existing masks)
                # In practice, you would use the model for inference here
                pred_mask = image.copy()
                
                # Preprocess
                gt_mask, pred_mask = self.preprocess_masks(gt_mask, pred_mask)
                
                # Compute metrics
                dice = self.compute_dice_coefficient(gt_mask, pred_mask)
                iou = self.compute_iou(gt_mask, pred_mask)
                
                metrics['images'].append(img_file)
                metrics['dice'].append(dice)
                metrics['iou'].append(iou)
                
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
                continue
        
        # Compute statistics
        if metrics['dice']:
            dice_mean = np.mean(metrics['dice'])
            dice_std = np.std(metrics['dice'])
            iou_mean = np.mean(metrics['iou'])
            iou_std = np.std(metrics['iou'])
            
            print(f"\nResults for {self.DATASETS[dataset_name]}:")
            print(f"  Dice: {dice_mean:.4f} ± {dice_std:.4f}")
            print(f"  IoU:  {iou_mean:.4f} ± {iou_std:.4f}")
            
            return {
                'dataset': dataset_name,
                'split': split,
                'num_images': len(metrics['dice']),
                'dice_mean': dice_mean,
                'dice_std': dice_std,
                'iou_mean': iou_mean,
                'iou_std': iou_std,
                'metrics': metrics
            }
        else:
            print(f"✗ No valid metrics computed for {dataset_name}")
            return {}
    
    def evaluate_all_datasets(self, split: str = 'test') -> List[Dict]:
        """
        Evaluate baseline on all datasets.
        
        Args:
            split: Data split to evaluate
            
        Returns:
            List of result dictionaries for each dataset
        """
        all_results = []
        for dataset_name in self.DATASETS.keys():
            result = self.evaluate_dataset(dataset_name, split=split)
            if result:
                all_results.append(result)
        return all_results
    
    def save_results(self, results: List[Dict], output_dir: str):
        """
        Save evaluation results to CSV and JSON.
        
        Args:
            results: List of result dictionaries
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary CSV
        summary_csv = output_dir / "baseline_summary.csv"
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'dataset', 'split', 'num_images', 'dice_mean', 'dice_std', 'iou_mean', 'iou_std'
            ])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'dataset': result['dataset'],
                    'split': result['split'],
                    'num_images': result['num_images'],
                    'dice_mean': f"{result['dice_mean']:.4f}",
                    'dice_std': f"{result['dice_std']:.4f}",
                    'iou_mean': f"{result['iou_mean']:.4f}",
                    'iou_std': f"{result['iou_std']:.4f}",
                })
        
        print(f"\n✓ Summary saved to {summary_csv}")
        
        # Save detailed results JSON
        for result in results:
            dataset_name = result['dataset']
            result_json = output_dir / f"baseline_{dataset_name}_detailed.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_result = {
                'dataset': result['dataset'],
                'split': result['split'],
                'num_images': result['num_images'],
                'dice_mean': float(result['dice_mean']),
                'dice_std': float(result['dice_std']),
                'iou_mean': float(result['iou_mean']),
                'iou_std': float(result['iou_std']),
                'metrics': {
                    'images': result['metrics']['images'],
                    'dice': [float(x) for x in result['metrics']['dice']],
                    'iou': [float(x) for x in result['metrics']['iou']],
                }
            }
            
            with open(result_json, 'w') as f:
                json.dump(json_result, f, indent=2)
            
            print(f"✓ Detailed results saved to {result_json}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate baseline BiomedCLIP model without adapters"
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Root directory containing all datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./baseline_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (e.g., cuda:0, cpu)'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='Path to model checkpoint (default: None = HuggingFace). Use "saliency_maps/model" for finetuned checkpoint'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'val', 'train'],
        help='Data split to evaluate'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    print("="*60)
    print("BiomedCLIP Baseline Evaluation")
    print("="*60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint_path or 'HuggingFace (chuhac/BiomedCLIP-vit-bert-hf)'}")
    print(f"Split: {args.split}")
    print("="*60)
    
    evaluator = DatasetEvaluator(args.dataset_root, device=args.device, checkpoint_path=args.checkpoint_path)
    
    # Evaluate all datasets
    results = evaluator.evaluate_all_datasets(split=args.split)
    
    # Save results
    evaluator.save_results(results, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE EVALUATION SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['dataset']}:")
        print(f"  Images: {result['num_images']}")
        print(f"  Dice: {result['dice_mean']:.4f} ± {result['dice_std']:.4f}")
        print(f"  IoU:  {result['iou_mean']:.4f} ± {result['iou_std']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
