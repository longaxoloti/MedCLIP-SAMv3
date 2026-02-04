#!/usr/bin/env python3
"""
Training Script for BiomedCLIP with Frequency Adapters

This script trains the frequency adapters on BiomedCLIP using DHN-NCE loss.

Usage:
    python train_frequency_adapter.py \
        --dataset-root /path/to/data \
        --dataset brain_tumors \
        --output-dir ./checkpoints \
        --epochs 50 \
        --batch-size 32
    
Example:
    python train_frequency_adapter.py \
        --dataset-root ~/projects/MedCLIP-SAMv3/data \
        --dataset brain_tumors \
        --output-dir ./checkpoints/brain_tumors \
        --epochs 50 \
        --batch-size 32 \
        --device cuda:0
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging
from torch.nn.utils.rnn import pad_sequence

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from biomedclip_finetuning.frequency_adapter import (
    inject_frequency_adapters,
    get_adapter_parameters,
    freeze_backbone_unfreeze_adapter
)

try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
except ImportError:
    print("Error: transformers library required. Install with: pip install transformers")
    sys.exit(1)

try:
    from loss.hnl import HardNegativeLoss
except ImportError:
    print("Warning: Could not import DHN-NCE loss, using simplified contrastive loss")
    HardNegativeLoss = None

import cv2
import json as json_lib


class MedicalImageTextDataset(Dataset):
    """Dataset for medical images with text prompts."""
    
    DATASET_CONFIGS = {
        'brain_tumors': {
            'prompts_path': 'saliency_maps/text_prompts/brain_tumors_training.json',
            'image_size': 224
        },
        'breast_tumors': {
            'prompts_path': 'saliency_maps/text_prompts/breast_tumors_training.json',
            'image_size': 224
        },
        'lung_CT': {
            'prompts_path': 'saliency_maps/text_prompts/lung_ct_training.json',
            'image_size': 224
        },
        'lung_Xray': {
            'prompts_path': 'saliency_maps/text_prompts/lung_xray_training.json',
            'image_size': 224
        },
    }
    
    def __init__(
        self, 
        dataset_path: str,
        dataset_name: str,
        split: str = 'train',
        processor = None,
        tokenizer = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            dataset_path: Path to dataset root directory
            dataset_name: Name of dataset (e.g., 'brain_tumors')
            split: Data split ('train', 'val')
            processor: Image processor for preprocessing
            tokenizer: Text tokenizer
            max_samples: Maximum number of samples to load
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.split = split
        self.processor = processor
        self.tokenizer = tokenizer
        
        # Get config
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.config = self.DATASET_CONFIGS[dataset_name]
        self.image_size = self.config['image_size']
        
        # Load image paths
        self.images_dir = self.dataset_path / dataset_name / f"{split}_images"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if max_samples:
            self.image_files = self.image_files[:max_samples]
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.images_dir}")
        
        print(f"✓ Loaded {len(self.image_files)} images from {split} split")
        
        # Load text prompts (use default prompts if file not found)
        self.text_prompts = self._load_prompts()
    
    def _load_prompts(self) -> List[str]:
        """Load text prompts for the dataset."""
        prompts_path = self.dataset_path.parent / self.config['prompts_path']
        
        if prompts_path.exists():
            try:
                with open(prompts_path, 'r') as f:
                    data = json_lib.load(f)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        # Extract prompts from dict if available
                        for key in ['prompts', 'texts', 'descriptions']:
                            if key in data:
                                return data[key]
                        # Otherwise return all string values
                        return [v for v in data.values() if isinstance(v, str)]
            except Exception as e:
                print(f"Warning: Could not load prompts from {prompts_path}: {e}")
        
        # Default prompts based on dataset
        default_prompts = self._get_default_prompts()
        print(f"Using {len(default_prompts)} default prompts for {self.dataset_name}")
        return default_prompts
    
    def _get_default_prompts(self) -> List[str]:
        """Get default text prompts for each dataset."""
        prompts = {
            'brain_tumors': [
                "A medical brain MRI scan showing a tumor.",
                "Brain tumor visible in the imaging.",
                "Abnormal brain tissue mass in MRI.",
                "Brain lesion in medical imaging.",
                "Tumor growth in brain tissue.",
            ],
            'breast_tumors': [
                "A medical breast mammogram showing a tumor.",
                "Breast tumor visible in the imaging.",
                "Abnormal breast tissue mass in mammography.",
                "Breast lesion in medical imaging.",
                "Tumor in breast tissue.",
            ],
            'lung_CT': [
                "A medical lung CT scan showing a lesion.",
                "Lung nodule visible in CT imaging.",
                "Abnormal lung tissue in CT scan.",
                "Lung lesion in medical imaging.",
                "Pulmonary abnormality in CT.",
            ],
            'lung_Xray': [
                "A medical chest X-ray showing a lesion.",
                "Lung abnormality visible in X-ray.",
                "Abnormal lung tissue in chest radiography.",
                "Lung lesion in chest X-ray.",
                "Pulmonary abnormality in radiography.",
            ],
        }
        return prompts.get(self.dataset_name, prompts['brain_tumors'])
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample.
        
        Args:
            idx: Index of sample
            
        Returns:
            Dictionary with 'image' and 'text' tensors
        """
        img_file = self.image_files[idx]
        img_path = self.images_dir / img_file
        
        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert grayscale to RGB (BiomedCLIP expects 3 channels)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Select random text prompt
        text = np.random.choice(self.text_prompts)
        
        # Process image + text jointly when possible
        if self.processor is not None:
            try:
                processed = self.processor(images=image, text=text, return_tensors='pt')
                image_tensor = processed['pixel_values'].squeeze(0)
                text_tokens = {
                    k: v.squeeze(0) for k, v in processed.items()
                    if k in ['input_ids', 'attention_mask', 'token_type_ids']
                }
            except Exception:
                # Fallback: process image and text separately
                image_resized = cv2.resize(image, (self.image_size, self.image_size))
                image_resized = image_resized.astype(np.float32) / 255.0
                image_t = torch.from_numpy(image_resized).permute(2, 0, 1)
                image_tensor = (image_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                               torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                
                if self.tokenizer is not None:
                    text_tokens = self.tokenizer([text], return_tensors='pt', padding=True)
                    text_tokens = {k: v.squeeze(0) for k, v in text_tokens.items()}
                else:
                    text_tokens = {
                        'input_ids': torch.zeros(1).long(),
                        'attention_mask': torch.ones(1).long()
                    }
        else:
            # Fallback: manual processing
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            image_tensor = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                           torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            if self.tokenizer is not None:
                text_tokens = self.tokenizer([text], return_tensors='pt', padding=True)
                text_tokens = {k: v.squeeze(0) for k, v in text_tokens.items()}
            else:
                text_tokens = {
                    'input_ids': torch.zeros(1).long(),
                    'attention_mask': torch.ones(1).long()
                }
        
        return {
            'image': image_tensor,
            'text': text_tokens,
            'text_str': text,
            'image_name': img_file,
        }


class MedpixDataset(Dataset):
    """Dataset for MedPix caption-based training (no segmentation masks)."""
    
    def __init__(
        self,
        csv_path: str,
        images_root: str,
        split: str = 'train',
        split_ratio: float = 0.85,
        seed: int = 42,
        processor = None,
        tokenizer = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize MedPix dataset with 85/15 train/val split.
        
        Args:
            csv_path: Path to medpix_dataset.csv
            images_root: Root directory for images
            split: Data split ('train' or 'val')
            split_ratio: Ratio for train/val split (default: 0.85)
            seed: Random seed for reproducibility
            processor: Image processor for preprocessing
            tokenizer: Text tokenizer
            max_samples: Maximum number of samples to load
        """
        self.csv_path = Path(csv_path)
        self.images_root = Path(images_root)
        self.split = split
        self.processor = processor
        self.tokenizer = tokenizer
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not self.images_root.exists():
            raise FileNotFoundError(f"Images root not found: {images_root}")
        
        # Load CSV
        import pandas as pd
        df = pd.read_csv(self.csv_path)
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Split data: 85% train, 15% val
        n_total = len(df)
        n_train = int(n_total * split_ratio)
        
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        if split == 'train':
            df = df.iloc[train_indices].reset_index(drop=True)
        else:  # val
            df = df.iloc[val_indices].reset_index(drop=True)
        
        # Store data (ensure caption is string)
        df['Caption'] = df['Caption'].fillna('').astype(str)
        self.captions = df['Caption'].tolist()
        self.filenames = df['filename'].tolist()
        
        # Filter valid samples (files that exist)
        valid_samples = []
        for caption, filename in zip(self.captions, self.filenames):
            caption = str(caption).strip()
            if not caption:
                continue
            # Extract just the filename (last part after /)
            # CSV contains full paths like: data/medpix_dataset/images/synpic100377.jpg
            # We only need: synpic100377.jpg
            just_filename = Path(filename).name
            
            img_path = self.images_root / just_filename
            if img_path.exists():
                valid_samples.append((caption, just_filename))
        
        if max_samples:
            valid_samples = valid_samples[:max_samples]
        
        if not valid_samples:
            raise ValueError(f"No valid samples found for {split} split")
        
        self.captions = [s[0] for s in valid_samples]
        self.filenames = [s[1] for s in valid_samples]
        
        print(f"✓ Loaded {len(self.captions)} {split} samples from MedPix")
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.captions)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample.
        
        Args:
            idx: Index of sample
            
        Returns:
            Dictionary with 'image' and 'text' tensors
        """
        caption = str(self.captions[idx]).strip()
        if not caption:
            caption = " "
        filename = self.filenames[idx]
        
        # Build image path
        img_path = self.images_root / filename
        
        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert grayscale to RGB (BiomedCLIP expects 3 channels)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image + text jointly when possible
        if self.processor is not None:
            try:
                processed = self.processor(images=image, text=caption, return_tensors='pt')
                image_tensor = processed['pixel_values'].squeeze(0)
                text_tokens = {
                    k: v.squeeze(0) for k, v in processed.items()
                    if k in ['input_ids', 'attention_mask', 'token_type_ids']
                }
            except Exception:
                # Fallback: process image and text separately
                image_resized = cv2.resize(image, (224, 224))
                image_resized = image_resized.astype(np.float32) / 255.0
                image_t = torch.from_numpy(image_resized).permute(2, 0, 1)
                image_tensor = (image_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                               torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                
                if self.tokenizer is not None:
                    text_tokens = self.tokenizer([caption], return_tensors='pt', padding=True)
                    text_tokens = {k: v.squeeze(0) for k, v in text_tokens.items()}
                else:
                    text_tokens = {
                        'input_ids': torch.zeros(1).long(),
                        'attention_mask': torch.ones(1).long()
                    }
        else:
            # Fallback: manual processing
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            image_tensor = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / \
                           torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            if self.tokenizer is not None:
                text_tokens = self.tokenizer([caption], return_tensors='pt', padding=True)
                text_tokens = {k: v.squeeze(0) for k, v in text_tokens.items()}
            else:
                text_tokens = {
                    'input_ids': torch.zeros(1).long(),
                    'attention_mask': torch.ones(1).long()
                }
        
        return {
            'image': image_tensor,
            'text': text_tokens,
            'text_str': caption,
            'image_name': filename,
        }


class FrequencyAdapterTrainer:
    """Trainer for frequency adapters on BiomedCLIP."""
    
    def __init__(
        self,
        dataset_name: str,
        output_dir: str,
        device: str = 'cuda:0',
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        adapter_layers: List[int] = None,
        rank_k: int = 64,
        gamma_init: float = 0.1,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            dataset_name: Name of dataset
            output_dir: Directory to save checkpoints
            device: Device for training
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            adapter_layers: List of layer indices to inject adapters
            rank_k: SVD rank for adapters
            gamma_init: Initial value for gamma (fusion weight)
            checkpoint_path: Path to BiomedCLIP checkpoint (None = HuggingFace, or custom path)
        """
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.adapter_layers = adapter_layers or [9, 10, 11]
        self.rank_k = rank_k
        self.gamma_init = gamma_init
        self.checkpoint_path = checkpoint_path
        
        # Setup logging
        self._setup_logging()
        
        # Load model
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load_model()
        
        # Setup loss
        self.loss_fn = HardNegativeLoss(temperature=0.07, beta1=1.0, beta2=1.0, alpha=0) \
                        if HardNegativeLoss else self._simple_contrastive_loss
        
        # Setup optimizer
        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer()
        
        self.logger.info(f"✓ Trainer initialized for {dataset_name}")
    
    def _setup_logging(self):
        """Setup logging."""
        log_file = self.output_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """Load BiomedCLIP model."""
        self.logger.info("Loading BiomedCLIP model...")
        
        # Default HuggingFace model
        hf_model_name = "chuhac/BiomedCLIP-vit-bert-hf"
        
        try:
            if self.checkpoint_path is None:
                # Load from HuggingFace Hub
                self.logger.info(f"  - Source: HuggingFace ({hf_model_name})")
                self.model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)
                self.processor = AutoProcessor.from_pretrained(hf_model_name, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
            else:
                # Load from local checkpoint
                self.logger.info(f"  - Source: Local checkpoint ({self.checkpoint_path})")
                self.model = AutoModel.from_pretrained(self.checkpoint_path, trust_remote_code=True)
                # Still use HF processor/tokenizer (they don't change)
                self.processor = AutoProcessor.from_pretrained(hf_model_name, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
            
            # Inject adapters
            self.logger.info(f"Injecting frequency adapters into layers {self.adapter_layers}...")
            self.model = inject_frequency_adapters(
                self.model,
                adapter_layers=self.adapter_layers,
                rank_k=self.rank_k,
                gamma_init=self.gamma_init
            )
            
            # Freeze backbone, unfreeze adapters
            freeze_backbone_unfreeze_adapter(self.model)
            
            self.model = self.model.to(self.device)
            self.model.eval()  # Will use train mode in training loop
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.logger.info(f"✓ Model loaded")
            self.logger.info(f"  - Total parameters: {total_params/1e6:.2f}M")
            self.logger.info(f"  - Trainable parameters: {trainable_params/1e6:.2f}M ({100*trainable_params/total_params:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Get adapter parameters only
        adapter_params = get_adapter_parameters(self.model)
        
        self.optimizer = AdamW(
            adapter_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        self.logger.info(f"✓ Optimizer setup")
        self.logger.info(f"  - Learning rate: {self.lr}")
        self.logger.info(f"  - Weight decay: {self.weight_decay}")
    
    @staticmethod
    def _simple_contrastive_loss(image_features, text_features, batch_size):
        """
        Simple contrastive loss as fallback when DHN-NCE is not available.
        
        Args:
            image_features: Image features (B, D)
            text_features: Text features (B, D)
            batch_size: Batch size
            
        Returns:
            Scalar loss
        """
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute similarity
        logits = torch.matmul(image_features, text_features.t()) / 0.07
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Compute cross-entropy loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2

    @staticmethod
    def _get_pooled_output(model_outputs):
        """Extract pooled output from transformer outputs (tuple or object)."""
        if hasattr(model_outputs, 'pooler_output') and model_outputs.pooler_output is not None:
            return model_outputs.pooler_output
        if isinstance(model_outputs, tuple) and len(model_outputs) > 1:
            return model_outputs[1]
        if hasattr(model_outputs, 'last_hidden_state'):
            return model_outputs.last_hidden_state[:, 0]
        if isinstance(model_outputs, tuple) and len(model_outputs) > 0:
            return model_outputs[0][:, 0]
        raise ValueError("Cannot extract pooled output from model outputs")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            
            # Get text features
            text_tokens = batch['text']
            if isinstance(text_tokens, torch.Tensor):
                text_tokens = {
                    'input_ids': text_tokens.to(self.device),
                    'attention_mask': torch.ones_like(text_tokens).to(self.device)
                }
            else:
                # text_tokens is already a dict
                text_tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in text_tokens.items()}
            
            # Forward pass (enable gradients for adapter training)
            image_outputs = self.model.vision_model(images)
            image_features = self._get_pooled_output(image_outputs)
            
            text_outputs = self.model.text_model(**text_tokens)
            text_features = self._get_pooled_output(text_outputs)
            
            # Compute loss
            batch_size = images.size(0)
            
            # Normalize features to prevent explosion
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            if isinstance(self.loss_fn, nn.Module):
                loss = self.loss_fn(image_features, text_features, batch_size)
            else:
                loss = self.loss_fn(image_features, text_features, batch_size)
            
            # Clamp loss to prevent NaN/Inf
            loss = torch.clamp(loss, min=-1e4, max=1e4)
            
            # Skip batch if loss is invalid
            if not torch.isfinite(loss):
                self.logger.warning(f"Invalid loss detected: {loss.item()}, skipping batch")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation statistics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            # Move to device
            images = batch['image'].to(self.device)
            
            # Get text features
            text_tokens = batch['text']
            if isinstance(text_tokens, torch.Tensor):
                text_tokens = {
                    'input_ids': text_tokens.to(self.device),
                    'attention_mask': torch.ones_like(text_tokens).to(self.device)
                }
            else:
                text_tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in text_tokens.items()}
            
            # Forward pass
            image_outputs = self.model.vision_model(images)
            image_features = self._get_pooled_output(image_outputs)
            
            text_outputs = self.model.text_model(**text_tokens)
            text_features = self._get_pooled_output(text_outputs)
            
            # Compute loss
            batch_size = images.size(0)
            
            # Normalize features to prevent explosion
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            if isinstance(self.loss_fn, nn.Module):
                loss = self.loss_fn(image_features, text_features, batch_size)
            else:
                loss = self.loss_fn(image_features, text_features, batch_size)
            
            # Clamp loss to prevent NaN/Inf
            loss = torch.clamp(loss, min=-1e4, max=1e4)
            
            # Skip batch if loss is invalid
            if not torch.isfinite(loss):
                continue
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'num_batches': num_batches
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """
        Save checkpoint.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary with metrics
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'dataset_name': self.dataset_name,
                'adapter_layers': self.adapter_layers,
                'rank_k': self.rank_k,
                'gamma_init': self.gamma_init,
                'lr': self.lr,
            }
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def train(
        self,
        dataset_root: str,
        num_epochs: int = 50,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        use_medpix: bool = True,
    ):
        """
        Train frequency adapters.
        
        Args:
            dataset_root: Root directory containing datasets
            num_epochs: Number of training epochs
            batch_size: Training batch size
            val_batch_size: Validation batch size (default: same as batch_size)
            num_workers: Number of data loading workers
            max_train_samples: Maximum training samples (for debugging)
            max_val_samples: Maximum validation samples (for debugging)
            use_medpix: Whether to use MedPix dataset (True) or medical segmentation datasets (False)
        """
        if val_batch_size is None:
            val_batch_size = batch_size
        
        self.logger.info("="*60)
        if use_medpix:
            self.logger.info("Training on MedPix dataset (85/15 split)")
        else:
            self.logger.info(f"Training on {self.dataset_name}")
        self.logger.info("="*60)
        self.logger.info(f"Epochs: {num_epochs}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("="*60)
        
        # Load datasets
        if use_medpix:
            # Use MedPix dataset (caption-based, no segmentation)
            # MedPix dataset is in same directory level as other datasets (data/medpix_dataset/)
            csv_path = Path(dataset_root) / 'medpix_dataset' / 'medpix_dataset.csv'
            images_root = Path(dataset_root) / 'medpix_dataset' / 'images'
            
            self.logger.info(f"Using MedPix CSV: {csv_path}")
            self.logger.info(f"Using MedPix images: {images_root}")
            
            # Check if paths exist
            if not csv_path.exists():
                self.logger.error(f"CSV not found at: {csv_path}")
                self.logger.error(f"Dataset root: {dataset_root}")
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            if not images_root.exists():
                self.logger.error(f"Images directory not found at: {images_root}")
                raise FileNotFoundError(f"Images directory not found: {images_root}")
            
            train_dataset = MedpixDataset(
                csv_path=str(csv_path),
                images_root=str(images_root),
                split='train',
                split_ratio=0.85,
                seed=42,
                processor=self.processor,
                tokenizer=self.tokenizer,
                max_samples=max_train_samples
            )
            
            val_dataset = MedpixDataset(
                csv_path=str(csv_path),
                images_root=str(images_root),
                split='val',
                split_ratio=0.85,
                seed=42,
                processor=self.processor,
                tokenizer=self.tokenizer,
                max_samples=max_val_samples
            )
        else:
            # Use medical segmentation datasets
            train_dataset = MedicalImageTextDataset(
                dataset_root,
                self.dataset_name,
                split='train',
                processor=self.processor,
                tokenizer=self.tokenizer,
                max_samples=max_train_samples
            )
            
            val_dataset = MedicalImageTextDataset(
                dataset_root,
                self.dataset_name,
                split='val',
                processor=self.processor,
                tokenizer=self.tokenizer,
                max_samples=max_val_samples
            )

        def collate_batch(batch: List[Dict]) -> Dict:
            """Custom collate to pad variable-length text tokens."""
            images = torch.stack([item['image'] for item in batch], dim=0)
            text_dicts = [item['text'] for item in batch]

            if isinstance(text_dicts[0], dict):
                input_ids = [td['input_ids'] for td in text_dicts]
                attention_mask = [td['attention_mask'] for td in text_dicts]

                padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
                padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

                text_tokens = {
                    'input_ids': padded_input_ids,
                    'attention_mask': padded_attention_mask
                }

                if 'token_type_ids' in text_dicts[0]:
                    token_type_ids = [td.get('token_type_ids', torch.zeros_like(td['input_ids'])) for td in text_dicts]
                    padded_token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
                    text_tokens['token_type_ids'] = padded_token_type_ids
            else:
                text_tokens = torch.stack(text_dicts, dim=0)

            return {
                'image': images,
                'text': text_tokens,
                'text_str': [item['text_str'] for item in batch],
                'image_name': [item['image_name'] for item in batch],
            }
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_batch
        )
        
        # Setup scheduler
        num_training_steps = len(train_loader) * num_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_training_steps)
        
        # Training loop
        best_val_loss = float('inf')
        training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
        }
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            self.logger.info(f"Train loss: {train_metrics['loss']:.4f}")
            self.logger.info(f"Val loss: {val_metrics['loss']:.4f}")
            
            # Save history
            training_history['epoch'].append(epoch + 1)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['val_loss'].append(val_metrics['loss'])
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, {
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
            })
            
            # Save best checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_checkpoint_path = self.output_dir / "checkpoint_best.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': {
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                    },
                    'config': {
                        'dataset_name': self.dataset_name,
                        'adapter_layers': self.adapter_layers,
                        'rank_k': self.rank_k,
                        'gamma_init': self.gamma_init,
                    }
                }, best_checkpoint_path)
                self.logger.info(f"✓ Best checkpoint saved: {best_checkpoint_path}")
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json_lib.dump(training_history, f, indent=2)
        
        self.logger.info(f"\n✓ Training complete!")
        self.logger.info(f"✓ Results saved to {self.output_dir}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train frequency adapters on BiomedCLIP"
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help='Root directory containing all datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='brain_tumors',
        choices=['brain_tumors', 'breast_tumors', 'lung_CT', 'lung_Xray'],
        help='Dataset to train on'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-6,
        help='Weight decay'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (e.g., cuda:0, cpu)'
    )
    parser.add_argument(
        '--adapter-layers',
        type=int,
        nargs='+',
        default=[9, 10, 11],
        help='Layer indices for adapter injection'
    )
    parser.add_argument(
        '--rank-k',
        type=int,
        default=64,
        help='SVD rank for adapters'
    )
    parser.add_argument(
        '--gamma-init',
        type=float,
        default=0.1,
        help='Initial gamma value for fusion weight'
    )
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        default=None,
        help='Path to BiomedCLIP checkpoint (default: None = HuggingFace). Use "saliency_maps/model" for finetuned checkpoint'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Max training samples (for debugging)'
    )
    parser.add_argument(
        '--max-val-samples',
        type=int,
        default=None,
        help='Max validation samples (for debugging)'
    )
    parser.add_argument(
        '--use-medpix',
        action='store_true',
        default=True,
        help='Use MedPix dataset (default: True). Use --no-medpix to use medical segmentation datasets'
    )
    parser.add_argument(
        '--no-medpix',
        dest='use_medpix',
        action='store_false',
        help='Use medical segmentation datasets instead of MedPix'
    )
    
    args = parser.parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.use_medpix:
        output_dir = os.path.join(args.output_dir, 'medpix', timestamp)
    else:
        output_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    
    # Create trainer
    trainer = FrequencyAdapterTrainer(
        dataset_name=args.dataset,
        output_dir=output_dir,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adapter_layers=args.adapter_layers,
        rank_k=args.rank_k,
        gamma_init=args.gamma_init,
        checkpoint_path=args.checkpoint_path,
    )
    
    # Start training
    trainer.train(
        dataset_root=args.dataset_root,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        use_medpix=args.use_medpix,
    )


if __name__ == "__main__":
    main()
