"""
Laplacian Pyramid Frequency Decomposition for ViT Patch Tokens

Adapts the Laplacian Pyramid approach from the SVD_Linear_Attention paper
to work on Vision Transformer patch tokens instead of raw images.

The paper: "Medical image segmentation based on frequency domain decomposition 
SVD linear attention" (Liu Qiong et al., 2025)

For ViT tokens with shape (B, 196, 768), we:
1. Reshape to spatial grid: (B, 768, 14, 14)
2. Apply Gaussian filters with different σ values
3. Compute Laplacian bands as differences
4. Return three frequency streams: Low, Mid-High, High
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LaplacianPyramidDecomposition(nn.Module):
    """
    Decomposes input tokens into frequency bands using Laplacian Pyramid.
    
    Args:
        d_model: Feature dimension (768 for BiomedCLIP)
        grid_size: Spatial grid size (14 for 14x14 patches)
        num_patches: Number of patches (196 = 14*14)
        sigmas: Gaussian filter standard deviations
        padding_mode: 'zeros', 'reflect', or 'replicate'
    """
    
    def __init__(
        self,
        d_model: int = 768,
        grid_size: int = 14,
        num_patches: int = 196,
        sigmas: Tuple[float, float, float, float] = (1.0, 3.0, 5.0, 7.0),
        padding_mode: str = "reflect"
    ):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.num_patches = num_patches
        self.sigmas = sigmas
        self.padding_mode = padding_mode
        
        # Create Gaussian kernels for different scales
        self.register_buffer('gaussian_kernels', self._create_gaussian_kernels())
    
    def _create_gaussian_kernels(self) -> torch.Tensor:
        """Create Gaussian filter kernels for each sigma."""
        kernels = []
        
        for sigma in self.sigmas:
            # For small grids (14x14), use smaller kernels
            # Kernel size capped at grid_size // 2
            max_kernel = min(int(4 * sigma + 1), self.grid_size - 2)
            max_kernel = max(3, max_kernel)
            
            # Ensure kernel size is odd
            kernel_size = max_kernel if max_kernel % 2 == 1 else max_kernel - 1
            kernel_size = max(3, kernel_size)
            
            # Create 1D Gaussian kernel
            ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
            g_x = torch.exp(-0.5 * (ax / sigma) ** 2)
            g_x = g_x / g_x.sum()  # Normalize
            
            # Create 2D Gaussian kernel
            g_kernel = g_x.unsqueeze(0) * g_x.unsqueeze(1)  # (K, K)
            g_kernel = g_kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
            
            kernels.append(g_kernel)
        
        # Stack kernels: pad all to largest kernel size
        max_kernel_size = max(k.shape[-1] for k in kernels)
        padded_kernels = []
        for k in kernels:
            if k.shape[-1] < max_kernel_size:
                # Pad kernel with zeros
                pad_size = (max_kernel_size - k.shape[-1]) // 2
                k = F.pad(k, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
            padded_kernels.append(k)
        
        return torch.cat(padded_kernels, dim=0)  # (4, 1, K_max, K_max)
    
    def _apply_gaussian_filter(self, x: torch.Tensor, kernel_idx: int) -> torch.Tensor:
        """
        Apply Gaussian filter with specific kernel.
        
        Args:
            x: Input tensor (B, C, H, W)
            kernel_idx: Index of kernel to use (0-3)
            
        Returns:
            Filtered tensor (B, C, H, W)
        """
        kernel = self.gaussian_kernels[kernel_idx:kernel_idx+1]  # (1, 1, K, K)
        
        # Expand kernel for each channel
        kernel = kernel.expand(x.shape[1], -1, -1, -1)  # (C, 1, K, K)
        kernel = kernel.to(x.device)
        
        # Apply padding - use reflection to avoid edge artifacts on small grids
        pad_size = kernel.shape[-1] // 2
        
        # Cap padding to reasonable size (max 3 for 14x14 grids)
        H, W = x.shape[-2:]
        pad_size = min(pad_size, (min(H, W) - 1) // 2, 3)
        
        if pad_size > 0:
            x = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Apply depthwise convolution
        output = F.conv2d(x, kernel, padding=0, groups=x.shape[1])
        
        # Resize if output spatial dim doesn't match input
        if output.shape[-2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output
    
    def forward(self, tokens: torch.Tensor, cls_token: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose tokens into frequency bands.
        
        Args:
            tokens: Patch tokens (B, 196, 768) or (B, 197, 768) with CLS
            cls_token: Optional CLS token to preserve (B, 1, 768)
            
        Returns:
            Tuple of (low_freq, mid_high_freq, high_freq), each (B, 196, 768) or (B, 197, 768)
        """
        batch_size = tokens.shape[0]
        
        # Handle CLS token if present
        has_cls = tokens.shape[1] == 197
        if has_cls:
            cls = tokens[:, 0:1, :]  # (B, 1, 768)
            patch_tokens = tokens[:, 1:, :]  # (B, 196, 768)
        else:
            cls = None
            patch_tokens = tokens  # (B, 196, 768)
        
        # Reshape patches to spatial grid: (B, 196, 768) -> (B, 768, 14, 14)
        spatial = patch_tokens.view(batch_size, self.grid_size, self.grid_size, self.d_model)
        spatial = spatial.permute(0, 3, 1, 2)  # (B, 768, 14, 14)
        
        # Apply Gaussian filters at different scales
        # σ in {1.0, 3.0, 5.0, 7.0}
        gaussian_1 = self._apply_gaussian_filter(spatial, 0)    # σ=1.0, detailed
        gaussian_2 = self._apply_gaussian_filter(spatial, 1)    # σ=3.0, medium
        gaussian_3 = self._apply_gaussian_filter(spatial, 2)    # σ=5.0, coarse
        gaussian_4 = self._apply_gaussian_filter(spatial, 3)    # σ=7.0, very coarse
        
        # Ensure all outputs have same spatial size (14, 14)
        target_size = (self.grid_size, self.grid_size)
        if gaussian_1.shape[-2:] != target_size:
            gaussian_1 = F.interpolate(gaussian_1, size=target_size, mode='bilinear', align_corners=False)
        if gaussian_2.shape[-2:] != target_size:
            gaussian_2 = F.interpolate(gaussian_2, size=target_size, mode='bilinear', align_corners=False)
        if gaussian_3.shape[-2:] != target_size:
            gaussian_3 = F.interpolate(gaussian_3, size=target_size, mode='bilinear', align_corners=False)
        if gaussian_4.shape[-2:] != target_size:
            gaussian_4 = F.interpolate(gaussian_4, size=target_size, mode='bilinear', align_corners=False)
        
        # Compute Laplacian bands (now all same size)
        # Band 1: Low frequency (very coarse structure)
        low_freq_spatial = gaussian_4
        
        # Band 2: Mid-High frequency (medium and local structure)
        # MidHigh = Gaussian(σ=3.0) - Gaussian(σ=5.0)
        mid_high_freq_spatial = gaussian_2 - gaussian_3
        
        # Band 3: High frequency (fine details and edges)
        # High = Gaussian(σ=1.0) - Gaussian(σ=3.0)
        high_freq_spatial = gaussian_1 - gaussian_2
        
        # Reshape back to tokens: (B, 768, 14, 14) -> (B, 196, 768)
        low_freq = low_freq_spatial.permute(0, 2, 3, 1).reshape(batch_size, -1, self.d_model)
        mid_high_freq = mid_high_freq_spatial.permute(0, 2, 3, 1).reshape(batch_size, -1, self.d_model)
        high_freq = high_freq_spatial.permute(0, 2, 3, 1).reshape(batch_size, -1, self.d_model)
        
        # Restore CLS token if it was present
        if has_cls:
            low_freq = torch.cat([cls, low_freq], dim=1)  # (B, 197, 768)
            mid_high_freq = torch.cat([cls, mid_high_freq], dim=1)
            high_freq = torch.cat([cls, high_freq], dim=1)
        
        return low_freq, mid_high_freq, high_freq
    
    def get_frequency_info(self, tokens: torch.Tensor) -> dict:
        """
        Get frequency information about the tokens for analysis.
        
        Returns:
            Dict with energy distribution across frequency bands
        """
        low_f, mid_f, high_f = self.forward(tokens)
        
        # Compute energy (L2 norm) in each band
        low_energy = torch.norm(low_f, p=2, dim=-1).mean()
        mid_energy = torch.norm(mid_f, p=2, dim=-1).mean()
        high_energy = torch.norm(high_f, p=2, dim=-1).mean()
        
        total_energy = low_energy + mid_energy + high_energy
        
        return {
            'low_freq_energy': low_energy.item(),
            'mid_high_freq_energy': mid_energy.item(),
            'high_freq_energy': high_energy.item(),
            'total_energy': total_energy.item(),
            'low_freq_ratio': (low_energy / total_energy).item() if total_energy > 0 else 0,
            'mid_high_freq_ratio': (mid_energy / total_energy).item() if total_energy > 0 else 0,
            'high_freq_ratio': (high_energy / total_energy).item() if total_energy > 0 else 0,
        }
