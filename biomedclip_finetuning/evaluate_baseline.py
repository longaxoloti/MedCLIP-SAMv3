#!/usr/bin/env python3
"""
Semantic Sanity Check for BiomedCLIP

This script validates that BiomedCLIP can be loaded and performs image-text encoding.
It computes semantic similarity between images and text prompts for diagnostic purposes.

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
    """Semantic evaluator for medical imaging datasets (sanity check only)."""
    
    DATASETS = {
        'brain_tumors': 'Brain Tumor',
        'breast_tumors': 'Breast Tumor',
        'lung_CT': 'Lung CT',
        'lung_Xray': 'Lung X-Ray'
    }
    
    # Default text prompts for semantic evaluation
    PROMPTS = {
        'brain_tumors': [
            "A medical brain MRI scan showing a tumor.",
            "Brain tumor visible in the imaging.",
            "Abnormal brain tissue mass in MRI.",
        ],
        'breast_tumors': [
            "A medical breast mammogram showing a tumor.",
            "Breast tumor visible in the imaging.",
            "Abnormal breast tissue mass.",
        ],
        'lung_CT': [
            "A medical lung CT scan showing a lesion.",
            "Lung nodule visible in CT imaging.",
            "Abnormal lung tissue in CT scan.",
        ],
        'lung_Xray': [
            "A medical chest X-ray showing a lesion.",
            "Lung abnormality visible in X-ray.",
            "Abnormal lung tissue in chest radiography.",
        ],
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
        
    def compute_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> float:
        """
        Compute cosine similarity between image and text features.
        
        Args:
            image_features: Image feature vector (D,)
            text_features: Text feature vector (D,)
            
        Returns:
            Similarity score in [0, 1]
        """
        from scipy.spatial.distance import cosine
        # cosine distance returns 0-1, convert to similarity
        similarity = 1.0 - cosine(image_features, text_features)
        return float(similarity)
    
    def evaluate_dataset(self, dataset_name: str, split: str = 'test', max_samples: int = 100) -> Dict:
        """
        Evaluate semantic similarity between images and prompts.
        
        Args:
            dataset_name: Name of dataset (e.g., 'brain_tumors')
            split: Data split ('test', 'val', or 'train')
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {self.DATASETS[dataset_name]} ({split})")
        print(f"{'='*60}")
        
        dataset_path = self.dataset_root / dataset_name
        images_dir = dataset_path / f"{split}_images"
        
        # Check if directory exists
        if not images_dir.exists():
            print(f"✗ Images directory not found: {images_dir}")
            return {}
        
        # Get list of images
        image_files = sorted([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not image_files:
            print(f"✗ No images found in {images_dir}")
            return {}
        
        # Limit to max_samples
        if len(image_files) > max_samples:
            image_files = image_files[:max_samples]
        
        print(f"Found {len(image_files)} images")
        
        # Get prompts for this dataset
        prompts = self.PROMPTS.get(dataset_name, self.PROMPTS['brain_tumors'])
        
        # Metrics storage
        metrics = {
            'images': [],
            'similarities': [],  # Store per-prompt similarities
        }
        
        # Evaluate each image
        for img_file in tqdm(image_files, desc="Evaluating"):
            img_path = images_dir / img_file
            
            try:
                # Load image
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"  Error: Could not load {img_file}")
                    continue
                
                # Convert grayscale to RGB
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] != 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Convert to tensor
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                image_tensor = image_tensor.to(self.baseline_model.device)
                
                # Get image features
                with torch.no_grad():
                    image_features = self.baseline_model.get_image_features(image_tensor)
                
                # Get text features for all prompts
                with torch.no_grad():
                    text_features = self.baseline_model.get_text_features(prompts)
                
                # Compute similarities
                similarities = []
                image_feat_np = image_features.cpu().numpy()[0]  # (D,)
                for i in range(len(prompts)):
                    text_feat_np = text_features.cpu().numpy()[i]  # (D,)
                    similarity = self.compute_similarity(image_feat_np, text_feat_np)
                    similarities.append(similarity)
                
                avg_similarity = np.mean(similarities)
                metrics['images'].append(img_file)
                metrics['similarities'].append(avg_similarity)
                
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
                continue
        
        # Compute statistics
        if metrics['similarities']:
            sim_mean = np.mean(metrics['similarities'])
            sim_std = np.std(metrics['similarities'])
            
            print(f"\nResults for {self.DATASETS[dataset_name]}:")
            print(f"  Avg Similarity: {sim_mean:.4f} ± {sim_std:.4f}")
            print(f"  (Lower values are typical; focus on consistency)")
            
            return {
                'dataset': dataset_name,
                'split': split,
                'num_images': len(metrics['similarities']),
                'similarity_mean': sim_mean,
                'similarity_std': sim_std,
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
        summary_csv = output_dir / "semantic_check_summary.csv"
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'dataset', 'split', 'num_images', 'similarity_mean', 'similarity_std'
            ])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'dataset': result['dataset'],
                    'split': result['split'],
                    'num_images': result['num_images'],
                    'similarity_mean': f"{result['similarity_mean']:.4f}",
                    'similarity_std': f"{result['similarity_std']:.4f}",
                })
        
        print(f"\n✓ Summary saved to {summary_csv}")
        
        # Save detailed results JSON
        for result in results:
            dataset_name = result['dataset']
            result_json = output_dir / f"semantic_check_{dataset_name}_detailed.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_result = {
                'dataset': result['dataset'],
                'split': result['split'],
                'num_images': result['num_images'],
                'similarity_mean': float(result['similarity_mean']),
                'similarity_std': float(result['similarity_std']),
                'metrics': {
                    'images': result['metrics']['images'],
                    'similarities': [float(x) for x in result['metrics']['similarities']],
                }
            }
            
            with open(result_json, 'w') as f:
                json.dump(json_result, f, indent=2)
            
            print(f"✓ Detailed results saved to {result_json}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Semantic sanity check for BiomedCLIP model"
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
        default='./semantic_check_results',
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
    parser.add_argument(
        '--max-samples',
        type=int,
        default=100,
        help='Maximum samples to evaluate per dataset'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    print("="*60)
    print("BiomedCLIP Semantic Sanity Check")
    print("="*60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint_path or 'HuggingFace (chuhac/BiomedCLIP-vit-bert-hf)'}")
    print(f"Split: {args.split}")
    print(f"Max samples: {args.max_samples}")
    print("="*60)
    print("NOTE: This is a semantic sanity check, not segmentation evaluation.")
    print("      For segmentation metrics, use postprocess_saliency_maps.py")
    print("="*60)
    
    evaluator = DatasetEvaluator(args.dataset_root, device=args.device, checkpoint_path=args.checkpoint_path)
    
    # Evaluate all datasets
    results = []
    for dataset_name in evaluator.DATASETS.keys():
        result = evaluator.evaluate_dataset(dataset_name, split=args.split, max_samples=args.max_samples)
        if result:
            results.append(result)
    
    # Save results
    evaluator.save_results(results, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SEMANTIC SANITY CHECK SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['dataset']}:")
        print(f"  Images evaluated: {result['num_images']}")
        print(f"  Avg similarity: {result['similarity_mean']:.4f} ± {result['similarity_std']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
