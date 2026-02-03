"""
Frequency-Adapter Module for BiomedCLIP
========================================

This package implements SVD-based Linear Attention adapters to enhance BiomedCLIP
with improved boundary and texture-aware feature extraction.

Key Components:
- SVD_Frequency_Adapter: Core low-rank attention mechanism
- BiomedCLIPEncoderLayerWithAdapter: Wrapper for side-car adapter integration
- inject_frequency_adapters: Utility to inject adapters into pre-trained BiomedCLIP

Usage:
    from frequency_adapter import inject_frequency_adapters
    
    model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf")
    model = inject_frequency_adapters(
        model, 
        adapter_layers=[9, 10, 11],  # Last 3 layers
        rank_k=64,
        gamma_init=0.1
    )
    
    # Train only adapter parameters
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'adapter' in name or 'gamma' in name:
            param.requires_grad = True
"""

from .svd_adapter import SVD_Frequency_Adapter
from .biomedclip_adapter import (
    BiomedCLIPEncoderLayerWithAdapter,
    inject_frequency_adapters,
    get_adapter_parameters,
    freeze_backbone_unfreeze_adapter
)

__all__ = [
    'SVD_Frequency_Adapter',
    'BiomedCLIPEncoderLayerWithAdapter',
    'inject_frequency_adapters',
    'get_adapter_parameters',
    'freeze_backbone_unfreeze_adapter',
]
