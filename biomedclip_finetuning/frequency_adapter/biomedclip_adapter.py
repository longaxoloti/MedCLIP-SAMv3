"""
BiomedCLIP Adapter Integration Module
======================================

Provides utilities to inject frequency adapters into pre-trained BiomedCLIP models
following the Side-car Adapter pattern. The core principle is:

    X_out = X_semantic + γ·X_freq + X_in
    
where:
    - X_semantic: Original frozen MHSA output (from BiomedCLIP)
    - X_freq: New learnable SVD-based attention output
    - γ: Learnable scale factor (initialized at 0.1)
    - X_in: Residual connection from input

This design preserves 100% of pre-trained knowledge while adding
frequency-aware feature extraction capabilities.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
import warnings

try:
    from .svd_adapter import SVD_Frequency_Adapter
except ImportError:
    from svd_adapter import SVD_Frequency_Adapter


class BiomedCLIPEncoderLayerWithAdapter(nn.Module):
    """
    Wrapper for BiomedCLIP encoder layer with integrated frequency adapter.
    
    This layer wraps the original BiomedCLIPEncoderLayer and adds a parallel
    SVD-based attention adapter. The adapter output is combined with the
    original semantic stream via a learnable scale factor γ.
    
    Architecture:
        Input X
        ├─→ [FROZEN] Original BiomedCLIP Layer → X_semantic
        ├─→ [TRAINABLE] SVD Adapter → X_freq  
        └─→ Fusion: X_out = X_semantic + γ·X_freq
    
    Parameters are frozen in the original layer and learnable in the adapter.
    The scale factor γ starts at 0.1 to ensure stable training without
    disrupting the semantic pathway.
    
    Args:
        original_layer: Pre-trained BiomedCLIPEncoderLayer (will be frozen)
        rank_k (int): Rank of SVD approximation in adapter (default: 64)
        gamma_init (float): Initial value for scale factor γ (default: 0.1)
        adapter_dropout (float): Dropout in adapter attention (default: 0.1)
        verbose (bool): Print layer information during init (default: False)
    
    Attributes:
        original_layer: Frozen semantic extraction branch
        adapter: SVD Linear Attention module
        gamma: Learnable scale factor for adapter contribution
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        rank_k: int = 64,
        gamma_init: float = 0.1,
        adapter_dropout: float = 0.1,
        verbose: bool = False
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.rank_k = rank_k
        self.gamma_init = gamma_init
        
        # Determine hidden dimension from original layer
        d_model = original_layer.embed_dim
        
        # Create adapter
        self.adapter = SVD_Frequency_Adapter(
            d_model=d_model,
            rank_k=rank_k,
            dropout=adapter_dropout
        )
        
        # Learnable scale factor (γ in the paper)
        # Start at small value to avoid disrupting semantic pathway
        self.gamma = nn.Parameter(torch.full((1,), gamma_init, dtype=torch.float32))
        
        # Freeze original layer completely
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        if verbose:
            total_params = sum(p.numel() for p in self.original_layer.parameters())
            adapter_params = sum(p.numel() for p in self.adapter.parameters())
            gamma_params = self.gamma.numel()
            print(
                f"Layer {self.__class__.__name__}: "
                f"frozen={total_params:,}, adapter={adapter_params:,}, γ={gamma_params}"
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with side-car adapter fusion.
        
        Args:
            hidden_states (torch.Tensor): Input features (B, N, D)
            attention_mask (torch.Tensor, optional): Attention mask
            output_attentions (bool): Return attention weights
        
        Returns:
            output (torch.Tensor): Fused features (B, N, D)
            attention_probs (torch.Tensor, optional): Attention weights if output_attentions=True
        
        Computation Flow:
            1. [FROZEN] Pass through original layer: X_semantic = original_layer(X_in)
            2. [TRAINABLE] Pass through adapter: X_freq = adapter(X_in)
            3. Fuse: X_out = X_semantic + γ·X_freq
        """
        # Step 1: Get semantic features from original frozen layer
        with torch.no_grad():
            semantic_output = self.original_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
        
        # semantic_output is a tuple: (hidden_states, [attention_probs])
        x_semantic = semantic_output[0]
        attention_probs = semantic_output[1] if output_attentions else None
        
        # Step 2: Get frequency features from trainable adapter
        x_freq = self.adapter(hidden_states)
        
        # Step 3: Fusion with learned scaling
        # γ·X_freq scales the contribution of frequency information
        output = x_semantic + self.gamma * x_freq
        
        # Prepare return tuple
        outputs = (output,)
        if output_attentions:
            outputs += (attention_probs,)
        
        return outputs
    
    def get_params_for_training(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """
        Separate trainable and frozen parameters.
        
        Returns:
            trainable (List[nn.Parameter]): Parameters to optimize
            frozen (List[nn.Parameter]): Parameters to keep frozen
        """
        trainable = []
        frozen = []
        
        # Adapter parameters
        for param in self.adapter.parameters():
            trainable.append(param)
        
        # Gamma parameter
        trainable.append(self.gamma)
        
        # Original layer parameters (all frozen)
        for param in self.original_layer.parameters():
            frozen.append(param)
        
        return trainable, frozen
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"d_model={self.original_layer.embed_dim}, "
            f"rank_k={self.rank_k}, "
            f"γ_init={self.gamma_init})"
        )


def inject_frequency_adapters(
    model: nn.Module,
    adapter_layers: Optional[List[int]] = None,
    rank_k: int = 64,
    gamma_init: float = 0.1,
    adapter_dropout: float = 0.1,
    verbose: bool = True
) -> nn.Module:
    """
    Inject frequency adapters into BiomedCLIP vision encoder.
    
    This function modifies the vision transformer encoder by replacing selected
    encoder layers with wrapped versions that include frequency adapters.
    
    Recommendation: Inject into last 3 layers (9, 10, 11) for optimal
    balance between capacity and training efficiency.
    
    Args:
        model: Pre-trained BiomedCLIP model (from HuggingFace)
        adapter_layers (List[int], optional): Indices of layers to inject adapters.
                                             Default: [9, 10, 11] (last 3 of 12 layers)
        rank_k (int): SVD rank for adapters (default: 64)
        gamma_init (float): Initial γ scale factor (default: 0.1)
        adapter_dropout (float): Dropout in adapters (default: 0.1)
        verbose (bool): Print injection progress (default: True)
    
    Returns:
        model: Modified model with adapters injected
    
    Notes:
        - Original layer parameters are FROZEN and not updated during training
        - Only adapter parameters and γ factors are trainable
        - Injection is done in-place; original model is modified
        - Use get_adapter_parameters() to get only trainable params for optimizer
    
    Example:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf")
        >>> model = inject_frequency_adapters(model, adapter_layers=[9, 10, 11])
        >>> optimizer = torch.optim.AdamW(get_adapter_parameters(model), lr=1e-4)
    """
    
    if adapter_layers is None:
        adapter_layers = [9, 10, 11]  # Last 3 of 12 layers
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Injecting Frequency Adapters into BiomedCLIP Vision Encoder")
        print(f"{'='*70}")
        print(f"Target layers: {adapter_layers}")
        print(f"SVD rank (k): {rank_k}")
        print(f"γ initialization: {gamma_init}")
        print(f"Adapter dropout: {adapter_dropout}\n")
    
    # Access vision encoder layers
    try:
        encoder_layers = model.vision_model.encoder.layers
        total_layers = len(encoder_layers)
    except AttributeError:
        raise ValueError(
            "Could not find vision_model.encoder.layers in the provided model. "
            "Ensure this is a valid BiomedCLIP model."
        )
    
    # Validate layer indices
    for layer_idx in adapter_layers:
        if not (0 <= layer_idx < total_layers):
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {total_layers-1}]"
            )
    
    # Inject adapters
    injected_count = 0
    for layer_idx in adapter_layers:
        original_layer = encoder_layers[layer_idx]
        
        wrapped_layer = BiomedCLIPEncoderLayerWithAdapter(
            original_layer=original_layer,
            rank_k=rank_k,
            gamma_init=gamma_init,
            adapter_dropout=adapter_dropout,
            verbose=False
        )
        
        encoder_layers[layer_idx] = wrapped_layer
        injected_count += 1
        
        if verbose:
            print(f"✓ Layer {layer_idx}: Adapter injected")
    
    if verbose:
        trainable_params = get_adapter_parameters(model)
        total_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n{'='*70}")
        print(f"Injection Summary:")
        print(f"  Total layers injected: {injected_count}/{total_layers}")
        print(f"  Trainable parameters: {total_trainable:,} ({100*total_trainable/total_params:.2f}%)")
        print(f"  Frozen parameters: {total_params - total_trainable:,} ({100*(total_params-total_trainable)/total_params:.2f}%)")
        print(f"{'='*70}\n")
    
    return model


def get_adapter_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Extract only adapter parameters (for optimizer).
    
    Returns:
        List of parameters that should be optimized during training
        (adapter weights + gamma factors)
    
    Usage:
        >>> adapter_params = get_adapter_parameters(model)
        >>> optimizer = torch.optim.AdamW(adapter_params, lr=1e-4)
    """
    params = []
    for name, param in model.named_parameters():
        if ('adapter' in name or 'gamma' in name) and param.requires_grad:
            params.append(param)
    return params


def freeze_backbone_unfreeze_adapter(model: nn.Module, verbose: bool = True) -> None:
    """
    Explicitly freeze BiomedCLIP backbone and unfreeze adapter parameters.
    
    This is a safety function to ensure proper parameter freezing.
    Should be called after model loading and adapter injection.
    
    Args:
        model: BiomedCLIP model with injected adapters
        verbose: Print freezing summary (default: True)
    """
    
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if 'adapter' in name or 'gamma' in name:
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    
    if verbose:
        print(f"\nParameter Freezing Summary:")
        print(f"  Frozen parameters: {frozen_count}")
        print(f"  Trainable parameters: {trainable_count}")
        print(f"  Total: {frozen_count + trainable_count}")


def verify_adapter_injection(model: nn.Module) -> Dict[str, Any]:
    """
    Verify that adapters were properly injected and configured.
    
    Returns:
        Dictionary with verification results
    """
    
    results = {
        'adapters_found': 0,
        'gammas_found': 0,
        'frozen_backbone': True,
        'trainable_adapters': True,
        'issues': []
    }
    
    for name, param in model.named_parameters():
        if 'adapter' in name:
            results['adapters_found'] += 1
            if param.requires_grad == False:
                results['issues'].append(f"Adapter parameter {name} is frozen (should be trainable)")
        
        if 'gamma' in name:
            results['gammas_found'] += 1
            if param.requires_grad == False:
                results['issues'].append(f"Gamma parameter {name} is frozen (should be trainable)")
        
        if 'adapter' not in name and 'gamma' not in name:
            if param.requires_grad == True:
                results['frozen_backbone'] = False
                results['issues'].append(f"Backbone parameter {name} is trainable (should be frozen)")
    
    return results
