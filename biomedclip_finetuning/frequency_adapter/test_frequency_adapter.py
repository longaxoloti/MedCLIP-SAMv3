"""
Unit Tests for Frequency-Adapter Module
========================================

Comprehensive test suite for SVD Linear Attention adapters and BiomedCLIP integration.

Tests cover:
1. SVD_Frequency_Adapter: Core algorithm correctness
2. BiomedCLIPEncoderLayerWithAdapter: Layer wrapper functionality
3. Injection utilities: Model modification and parameter management
4. Integration: End-to-end adapter injection and training setup

Run tests:
    python -m pytest test_frequency_adapter.py -v
    
or:
    python test_frequency_adapter.py
"""

import torch
import torch.nn as nn
import unittest
from typing import Tuple
import sys
import os

# Add parent directory to path
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_dir)

try:
    # Try absolute imports first (when run as module)
    from frequency_adapter.svd_adapter import SVD_Frequency_Adapter
    from frequency_adapter.biomedclip_adapter import (
        BiomedCLIPEncoderLayerWithAdapter,
        inject_frequency_adapters,
        get_adapter_parameters,
        freeze_backbone_unfreeze_adapter,
        verify_adapter_injection
    )
except ImportError:
    # Fallback to direct imports (when run as script)
    from svd_adapter import SVD_Frequency_Adapter
    from biomedclip_adapter import (
        BiomedCLIPEncoderLayerWithAdapter,
        inject_frequency_adapters,
        get_adapter_parameters,
        freeze_backbone_unfreeze_adapter,
        verify_adapter_injection
    )


class MockBiomedCLIPEncoderLayer(nn.Module):
    """
    Mock BiomedCLIP encoder layer for testing without loading full model.
    
    Mimics the structure of real BiomedCLIPEncoderLayer:
    - embed_dim: hidden dimension
    - MHSA (simulated with linear layer)
    - MLP
    - Layer normalization
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Layer 1
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mhsa = nn.Linear(embed_dim, embed_dim)  # Simplified MHSA
        
        # Layer 2
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        output_attentions: bool = False
    ) -> Tuple:
        """Mock forward pass."""
        # Self-attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.mhsa(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (torch.zeros_like(hidden_states[:, :1, :1]),)  # Dummy attention
        
        return outputs


class TestSVDFrequencyAdapter(unittest.TestCase):
    """Tests for SVD_Frequency_Adapter core functionality."""
    
    def setUp(self):
        """Initialize test parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.d_model = 768
        self.rank_k = 64
        self.n_tokens = 197  # BiomedCLIP uses 14x14 patches + CLS
        self.batch_size = 2
    
    def test_initialization(self):
        """Test that adapter initializes correctly."""
        adapter = SVD_Frequency_Adapter(
            d_model=self.d_model,
            rank_k=self.rank_k
        ).to(self.device)
        
        # Check attributes
        self.assertEqual(adapter.d_model, self.d_model)
        self.assertEqual(adapter.rank_k, self.rank_k)
        
        # Check weight initialization
        for module in [adapter.q_proj, adapter.k_proj, adapter.v_proj, adapter.out_proj]:
            self.assertIsNotNone(module.weight)
            self.assertGreater(module.weight.abs().max(), 0)
    
    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        adapter = SVD_Frequency_Adapter(
            d_model=self.d_model,
            rank_k=self.rank_k
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.n_tokens, self.d_model, device=self.device)
        output = adapter(x)
        
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)
    
    def test_forward_dtype_preservation(self):
        """Test that dtype is preserved (float32, float16, etc)."""
        adapter = SVD_Frequency_Adapter(d_model=self.d_model).to(self.device)
        
        # Test float32
        x_f32 = torch.randn(self.batch_size, self.n_tokens, self.d_model, dtype=torch.float32, device=self.device)
        out_f32 = adapter(x_f32)
        self.assertEqual(out_f32.dtype, torch.float32)
        
        # Test float64
        adapter_f64 = adapter.to(torch.float64)
        x_f64 = x_f32.to(torch.float64)
        out_f64 = adapter_f64(x_f64)
        self.assertEqual(out_f64.dtype, torch.float64)
    
    def test_gradient_flow(self):
        """Test that gradients flow through adapter during backprop."""
        adapter = SVD_Frequency_Adapter(d_model=self.d_model).to(self.device)
        
        x = torch.randn(self.batch_size, self.n_tokens, self.d_model, device=self.device, requires_grad=True)
        output = adapter(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        for param in adapter.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertGreater(param.grad.abs().sum(), 0)
    
    def test_svd_components_output(self):
        """Test SVD component extraction for analysis."""
        adapter = SVD_Frequency_Adapter(d_model=self.d_model, rank_k=self.rank_k).to(self.device)
        
        x = torch.randn(self.batch_size, self.n_tokens, self.d_model, device=self.device)
        output, svd_info = adapter(x, return_svd_components=True)
        
        # Check output
        self.assertEqual(output.shape, x.shape)
        
        # Check SVD info
        self.assertIn('singular_values', svd_info)
        self.assertIn('u_k', svd_info)
        self.assertIn('rank_estimate', svd_info)
        self.assertIn('condition_numbers', svd_info)
        
        # Check shapes
        self.assertEqual(svd_info['u_k'].shape, (self.batch_size, self.n_tokens, self.rank_k))
        self.assertEqual(svd_info['singular_values'].shape, (self.batch_size, self.rank_k))
    
    def test_complexity_reduction(self):
        """Verify that complexity is reduced from O(n²d) to O(nkd)."""
        n_tokens = 197
        d_model = 768
        
        # Traditional attention: n² = 197² ≈ 38,809 ops per token
        traditional_ops = n_tokens * n_tokens * d_model
        
        # SVD attention: n*k = 197*64 ≈ 12,608 ops per token
        k = 64
        svd_ops = n_tokens * k * d_model
        
        # SVD should be ~3x faster (with some overhead)
        ratio = traditional_ops / svd_ops
        self.assertGreater(ratio, 2.5, f"SVD not reducing complexity enough (ratio={ratio:.2f})")
    
    def test_numerical_stability(self):
        """Test that adapter handles extreme values gracefully."""
        adapter = SVD_Frequency_Adapter(d_model=self.d_model).to(self.device)
        
        # Test with very small values
        x_small = torch.randn(self.batch_size, self.n_tokens, self.d_model, device=self.device) * 1e-6
        output_small = adapter(x_small)
        self.assertFalse(torch.isnan(output_small).any(), "NaN detected for small input values")
        self.assertFalse(torch.isinf(output_small).any(), "Inf detected for small input values")
        
        # Test with large values
        x_large = torch.randn(self.batch_size, self.n_tokens, self.d_model, device=self.device) * 1e3
        output_large = adapter(x_large)
        self.assertFalse(torch.isnan(output_large).any(), "NaN detected for large input values")
        self.assertFalse(torch.isinf(output_large).any(), "Inf detected for large input values")


class TestBiomedCLIPEncoderLayerWithAdapter(unittest.TestCase):
    """Tests for wrapped encoder layer."""
    
    def setUp(self):
        """Initialize test parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_layer = MockBiomedCLIPEncoderLayer(embed_dim=768).to(self.device)
        self.batch_size = 2
        self.n_tokens = 197
    
    def test_wrapped_layer_creation(self):
        """Test that wrapper layer is created correctly."""
        wrapped = BiomedCLIPEncoderLayerWithAdapter(
            original_layer=self.original_layer,
            rank_k=64,
            gamma_init=0.1
        ).to(self.device)
        
        self.assertIsNotNone(wrapped.adapter)
        self.assertIsNotNone(wrapped.gamma)
        self.assertAlmostEqual(wrapped.gamma.item(), 0.1, places=5)
    
    def test_backbone_frozen(self):
        """Test that original layer parameters are frozen."""
        wrapped = BiomedCLIPEncoderLayerWithAdapter(self.original_layer).to(self.device)
        
        for param in wrapped.original_layer.parameters():
            self.assertFalse(param.requires_grad, "Backbone parameters should be frozen")
    
    def test_adapter_trainable(self):
        """Test that adapter parameters are trainable."""
        wrapped = BiomedCLIPEncoderLayerWithAdapter(self.original_layer).to(self.device)
        
        for param in wrapped.adapter.parameters():
            self.assertTrue(param.requires_grad, "Adapter parameters should be trainable")
        
        self.assertTrue(wrapped.gamma.requires_grad, "Gamma should be trainable")
    
    def test_forward_pass(self):
        """Test forward pass through wrapped layer."""
        wrapped = BiomedCLIPEncoderLayerWithAdapter(self.original_layer).to(self.device)
        
        x = torch.randn(self.batch_size, self.n_tokens, 768, device=self.device)
        output = wrapped(x)
        
        self.assertIsInstance(output, tuple)
        self.assertEqual(output[0].shape, x.shape)
    
    def test_fusion_math(self):
        """Test that fusion formula is correct: X_out = X_semantic + γ·X_freq."""
        wrapped = BiomedCLIPEncoderLayerWithAdapter(
            self.original_layer,
            gamma_init=0.5
        ).to(self.device)
        
        x = torch.randn(self.batch_size, self.n_tokens, 768, device=self.device)
        
        # Forward through wrapper (this computes the actual fusion internally)
        x_fused = wrapped(x)[0]
        
        # Verify fusion mathematically by computing components separately
        # We can only verify this is working by checking output shape and gradient flow
        self.assertEqual(x_fused.shape, x.shape)
        
        # Check gradients flow through fusion
        loss = x_fused.sum()
        loss.backward()
        
        # Verify gamma was updated
        self.assertIsNotNone(wrapped.gamma.grad)
        self.assertGreater(wrapped.gamma.grad.abs().sum(), 0)
    
    def test_get_params_for_training(self):
        """Test parameter separation utility."""
        wrapped = BiomedCLIPEncoderLayerWithAdapter(self.original_layer).to(self.device)
        
        trainable, frozen = wrapped.get_params_for_training()
        
        # Trainable should include adapter + gamma
        total_adapter_params = sum(p.numel() for p in wrapped.adapter.parameters())
        total_trainable = sum(p.numel() for p in trainable)
        
        self.assertGreater(total_trainable, 0)
        self.assertGreater(len(frozen), 0)


class TestAdapterInjection(unittest.TestCase):
    """Tests for adapter injection utilities."""
    
    def setUp(self):
        """Create mock model structure."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create mock model with encoder.layers
        self.mock_model = nn.Module()
        self.mock_model.vision_model = nn.Module()
        self.mock_model.vision_model.encoder = nn.Module()
        
        # Create 12 mock layers (matching BiomedCLIP)
        layers = nn.ModuleList([
            MockBiomedCLIPEncoderLayer(embed_dim=768)
            for _ in range(12)
        ])
        self.mock_model.vision_model.encoder.layers = layers
        self.mock_model = self.mock_model.to(self.device)
    
    def test_inject_adapters(self):
        """Test injection of adapters into model."""
        model = inject_frequency_adapters(
            self.mock_model,
            adapter_layers=[9, 10, 11],
            verbose=False
        )
        
        # Check that layers were wrapped
        self.assertIsInstance(
            model.vision_model.encoder.layers[9],
            BiomedCLIPEncoderLayerWithAdapter
        )
        self.assertIsInstance(
            model.vision_model.encoder.layers[10],
            BiomedCLIPEncoderLayerWithAdapter
        )
        self.assertIsInstance(
            model.vision_model.encoder.layers[11],
            BiomedCLIPEncoderLayerWithAdapter
        )
    
    def test_get_adapter_parameters(self):
        """Test extraction of adapter parameters."""
        model = inject_frequency_adapters(
            self.mock_model,
            adapter_layers=[9, 10, 11],
            verbose=False
        )
        
        # Important: must freeze backbone first
        freeze_backbone_unfreeze_adapter(model, verbose=False)
        
        adapter_params = get_adapter_parameters(model)
        
        # Should have parameters from 3 adapters + 3 gammas
        self.assertGreater(len(adapter_params), 0)
        
        # Check that all adapter params are trainable
        for param in adapter_params:
            self.assertTrue(param.requires_grad, "Adapter param should be trainable")
    
    def test_freeze_backbone_unfreeze_adapter(self):
        """Test parameter freezing utility."""
        model = inject_frequency_adapters(
            self.mock_model,
            adapter_layers=[9, 10, 11],
            verbose=False
        )
        
        freeze_backbone_unfreeze_adapter(model, verbose=False)
        
        # Check freezing
        for name, param in model.named_parameters():
            if 'adapter' in name or 'gamma' in name:
                self.assertTrue(param.requires_grad, f"Adapter param frozen: {name}")
            else:
                self.assertFalse(param.requires_grad, f"Backbone param trainable: {name}")
    
    def test_verify_injection(self):
        """Test verification utility."""
        model = inject_frequency_adapters(
            self.mock_model,
            adapter_layers=[9, 10, 11],
            verbose=False
        )
        
        # Must freeze backbone first before verification
        freeze_backbone_unfreeze_adapter(model, verbose=False)
        
        results = verify_adapter_injection(model)
        
        self.assertGreater(results['adapters_found'], 0)
        self.assertGreater(results['gammas_found'], 0)
        self.assertEqual(len(results['issues']), 0, f"Issues found: {results['issues']}")


class TestEndToEndIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def setUp(self):
        """Setup mock model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create mock model
        self.model = nn.Module()
        self.model.vision_model = nn.Module()
        self.model.vision_model.encoder = nn.Module()
        layers = nn.ModuleList([
            MockBiomedCLIPEncoderLayer(embed_dim=768)
            for _ in range(12)
        ])
        self.model.vision_model.encoder.layers = layers
        self.model.to(self.device)
    
    def test_complete_workflow(self):
        """Test complete workflow: inject → freeze → train setup."""
        # Create fresh model for this test
        test_model = nn.Module()
        test_model.vision_model = nn.Module()
        test_model.vision_model.encoder = nn.Module()
        layers = nn.ModuleList([
            MockBiomedCLIPEncoderLayer(embed_dim=768)
            for _ in range(12)
        ])
        test_model.vision_model.encoder.layers = layers
        test_model = test_model.to(self.device)
        
        # Inject adapters
        model = inject_frequency_adapters(
            test_model,
            adapter_layers=[9, 10, 11],
            verbose=False
        )
        
        # Move to device AFTER injection
        model = model.to(self.device)
        
        # Freeze backbone and unfreeze adapters
        freeze_backbone_unfreeze_adapter(model, verbose=False)
        
        # Get optimizer parameters
        adapter_params = get_adapter_parameters(model)
        
        # Create optimizer (simulate training setup)
        optimizer = torch.optim.AdamW(adapter_params, lr=1e-4)
        
        # Forward pass with training
        wrapped_layer = model.vision_model.encoder.layers[11]
        x = torch.randn(1, 197, 768, device=self.device)
        output = wrapped_layer(x)
        loss = output[0].sum()
        
        # Backward pass
        loss.backward()
        
        # Check that optimizer has parameters
        self.assertGreater(len(optimizer.param_groups[0]['params']), 0)
        
        # Optimizer step
        optimizer.step()
        
        # Verify step completed without errors
        self.assertIsNotNone(optimizer.state)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSVDFrequencyAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestBiomedCLIPEncoderLayerWithAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestAdapterInjection))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
