"""
Ablation Testing Infrastructure for Frequency Adapter V2

This module provides comprehensive tests for each component of the redesigned
frequency adapter to verify correctness, numerical stability, and contribution
to segmentation performance.

Tests include:
1. Laplacian Pyramid decomposition energy distribution
2. SVD adapter output norms and stability
3. Gamma clamping effectiveness  
4. Multi-stream fusion weights normalization
5. Backward pass gradient flow
6. Numerical stability with extreme inputs
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AblationTester:
    """Comprehensive ablation testing for frequency adapter components."""
    
    def __init__(self, device: str = 'cuda:0'):
        self.device = torch.device(device)
    
    def test_laplacian_pyramid(self) -> Dict[str, float]:
        """
        Test Laplacian Pyramid decomposition.
        
        Verifies:
        - Three frequency bands are created
        - Energy is distributed across bands
        - Values are numerically stable (no NaN/Inf)
        
        Returns:
            Dictionary with test results
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Laplacian Pyramid Decomposition")
        logger.info("="*70)
        
        try:
            from biomedclip_finetuning.frequency_adapter.laplacian_decomposition import LaplacianPyramidDecomposition
        except ImportError as e:
            logger.error(f"Could not import LaplacianPyramidDecomposition: {e}")
            return {'status': 'FAILED', 'error': str(e)}
        
        results = {}
        
        try:
            # Create module
            laplacian = LaplacianPyramidDecomposition(
                d_model=768,
                grid_size=14,
                num_patches=196
            ).to(self.device)
            
            # Create input with CLS token (B, 197, 768)
            x = torch.randn(2, 197, 768, device=self.device)
            
            # Forward pass
            low_f, mid_f, high_f = laplacian(x)
            
            # Check shapes
            assert low_f.shape == x.shape, f"Low freq shape {low_f.shape} != input {x.shape}"
            assert mid_f.shape == x.shape, f"Mid-high freq shape {mid_f.shape} != input {x.shape}"
            assert high_f.shape == x.shape, f"High freq shape {high_f.shape} != input {x.shape}"
            logger.info("✓ All frequency bands have correct shape: (B, 197, 768)")
            results['shape_correct'] = True
            
            # Check for numerical stability
            for band, name in [(low_f, 'Low'), (mid_f, 'Mid-High'), (high_f, 'High')]:
                has_nan = torch.isnan(band).any().item()
                has_inf = torch.isinf(band).any().item()
                assert not has_nan, f"{name}-frequency band contains NaN"
                assert not has_inf, f"{name}-frequency band contains Inf"
            logger.info("✓ No NaN/Inf in any frequency band")
            results['no_nan_inf'] = True
            
            # Check energy distribution
            freq_info = laplacian.get_frequency_info(x)
            logger.info(f"\nFrequency Energy Distribution:")
            logger.info(f"  Low-frequency:      {freq_info['low_freq_ratio']*100:6.2f}%")
            logger.info(f"  Mid-High-frequency: {freq_info['mid_high_freq_ratio']*100:6.2f}%")
            logger.info(f"  High-frequency:     {freq_info['high_freq_ratio']*100:6.2f}%")
            
            # Verify energy distribution is reasonable (no single band dominates)
            assert freq_info['low_freq_ratio'] < 0.8, "Low-freq should not dominate"
            assert freq_info['high_freq_ratio'] > 0.0, "High-freq should have some energy"
            logger.info("✓ Energy distribution is reasonable")
            results['energy_distribution'] = True
            
            results['status'] = 'PASSED'
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
        
        return results
    
    def test_gamma_clamping(self) -> Dict[str, float]:
        """
        Test gamma clamping mechanism.
        
        Verifies:
        - Gamma values are clamped to [0, 0.5]
        - Clamping prevents unbounded growth
        - Contribution ratio stays reasonable
        
        Returns:
            Dictionary with test results
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Gamma Clamping Mechanism")
        logger.info("="*70)
        
        try:
            from biomedclip_finetuning.frequency_adapter.biomedclip_adapter import BiomedCLIPEncoderLayerWithAdapter
        except ImportError as e:
            logger.error(f"Could not import BiomedCLIPEncoderLayerWithAdapter: {e}")
            return {'status': 'FAILED', 'error': str(e)}
        
        results = {}
        
        try:
            # Create mock layer
            class MockLayer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed_dim = 768
                
                def forward(self, hidden_states, attention_mask=None, output_attentions=False):
                    return (hidden_states,)
            
            # Create wrapped layer with adapter
            wrapped = BiomedCLIPEncoderLayerWithAdapter(
                original_layer=MockLayer().to(self.device),
                rank_k=64,
                gamma_init=0.1
            ).to(self.device)
            
            # Test 1: Check initial gamma
            initial_gamma = wrapped.gamma.item()
            assert abs(initial_gamma - 0.1) < 1e-4, f"Initial gamma should be ~0.1, got {initial_gamma}"
            logger.info(f"✓ Initial gamma value: {initial_gamma:.4f}")
            results['initial_gamma'] = initial_gamma
            
            # Test 2: Simulate gamma growth during training
            x = torch.randn(2, 197, 768, device=self.device)
            
            # Simulate multiple training steps that would grow gamma
            for step in range(5):
                wrapped.gamma.data += torch.tensor(0.3, device=self.device)  # Simulate growth
            
            grown_gamma = wrapped.gamma.item()
            logger.info(f"✓ Gamma after simulated growth: {grown_gamma:.4f}")
            results['grown_gamma'] = grown_gamma
            
            # Test 3: Check clamping in forward pass
            y = wrapped(x)
            
            # Get feature norms
            norms = wrapped.get_feature_norms(x)
            logger.info(f"\nFeature norms with gamma={grown_gamma:.4f}:")
            logger.info(f"  Semantic norm:    {norms['semantic_norm']:.2f}")
            logger.info(f"  Frequency norm:   {norms['freq_norm']:.2f}")
            logger.info(f"  Clamped gamma:    {norms['gamma_clamped']:.4f}")
            logger.info(f"  Contribution %:   {norms['contribution_ratio']*100:.2f}%")
            
            # Verify clamping works
            assert norms['gamma_clamped'] <= 0.5, f"Gamma should be clamped to <=0.5"
            logger.info("✓ Gamma successfully clamped to max 0.5")
            results['gamma_clamped'] = norms['gamma_clamped']
            
            # Verify contribution is reasonable (< 50%)
            assert norms['contribution_ratio'] < 0.5, "Adapter should not dominate (< 50%)"
            logger.info("✓ Adapter contribution ratio < 50% (adapter doesn't overwhelm semantic)")
            results['contribution_ratio'] = norms['contribution_ratio']
            
            results['status'] = 'PASSED'
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
        
        return results
    
    def test_multi_stream_fusion(self) -> Dict[str, float]:
        """
        Test multi-stream fusion weights.
        
        Verifies:
        - Fusion weights sum to 1.0 (normalized)
        - All three streams are active
        - Weights are learnable
        
        Returns:
            Dictionary with test results
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Multi-Stream Fusion Weights")
        logger.info("="*70)
        
        try:
            from biomedclip_finetuning.frequency_adapter.svd_adapter import SVDFrequencyAdapterMultiStream
        except ImportError as e:
            logger.error(f"Could not import SVDFrequencyAdapterMultiStream: {e}")
            return {'status': 'FAILED', 'error': str(e)}
        
        results = {}
        
        try:
            # Create adapter
            adapter = SVDFrequencyAdapterMultiStream(d_model=768).to(self.device)
            
            # Create input
            x = torch.randn(2, 197, 768, device=self.device)
            x.requires_grad = True
            
            # Forward pass with frequency stats
            y, freq_stats = adapter(x, return_freq_stats=True)
            
            # Get fusion weights
            fusion_w = torch.nn.functional.softmax(adapter.fusion_weights, dim=0)
            fusion_w_vals = fusion_w.detach().cpu().numpy()
            
            logger.info(f"\nFusion weights (after softmax):")
            logger.info(f"  Low-frequency:      {fusion_w_vals[0]:.4f}")
            logger.info(f"  Mid-High-frequency: {fusion_w_vals[1]:.4f}")
            logger.info(f"  High-frequency:     {fusion_w_vals[2]:.4f}")
            logger.info(f"  Sum:                {fusion_w_vals.sum():.4f}")
            
            # Verify normalization
            assert np.isclose(fusion_w_vals.sum(), 1.0), "Fusion weights should sum to 1.0"
            logger.info("✓ Fusion weights sum to 1.0 (properly normalized)")
            results['weights_normalized'] = True
            
            # Verify all streams are active
            for i, w in enumerate(fusion_w_vals):
                assert w > 0.0, f"Stream {i} has weight 0 (should be > 0)"
                assert w < 1.0, f"Stream {i} dominates (weight={w:.4f} >= 1.0)"
            logger.info("✓ All three frequency streams are active (0 < weight < 1 for each)")
            results['all_streams_active'] = True
            
            # Check that fusion weights are learnable
            assert adapter.fusion_weights.requires_grad, "Fusion weights should be learnable"
            logger.info("✓ Fusion weights are learnable parameters")
            results['learnable'] = True
            
            # Test backward pass
            loss = y.mean()
            loss.backward()
            assert adapter.fusion_weights.grad is not None, "Gradients should flow to fusion weights"
            logger.info("✓ Gradients successfully flow to fusion weights")
            results['gradients_flow'] = True
            
            results['status'] = 'PASSED'
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
        
        return results
    
    def test_backward_pass(self) -> Dict[str, float]:
        """
        Test backward pass and gradient flow.
        
        Verifies:
        - Gradients flow to all adapter parameters
        - No gradient explosion/vanishing
        - Numerical stability during backprop
        
        Returns:
            Dictionary with test results
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Backward Pass and Gradient Flow")
        logger.info("="*70)
        
        try:
            from biomedclip_finetuning.frequency_adapter.svd_adapter import SVD_Frequency_Adapter
        except ImportError as e:
            logger.error(f"Could not import SVD_Frequency_Adapter: {e}")
            return {'status': 'FAILED', 'error': str(e)}
        
        results = {}
        
        try:
            # Create adapter
            adapter = SVD_Frequency_Adapter(d_model=768, rank_k=64).to(self.device)
            
            # Create input
            x = torch.randn(2, 197, 768, device=self.device, requires_grad=True)
            
            # Forward pass
            y, svd_info = adapter(x, return_svd_components=True)
            
            # Compute loss
            loss = y.mean()
            
            # Backward pass
            loss.backward()
            
            # Check gradient flow
            gradient_stats = {}
            for name, param in adapter.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_stats[name] = grad_norm
                    
                    if np.isnan(grad_norm) or np.isinf(grad_norm):
                        logger.error(f"✗ Invalid gradient for {name}: {grad_norm}")
                        results['status'] = 'FAILED'
                        return results
            
            logger.info(f"\nGradient norms for parameters:")
            for name, grad_norm in gradient_stats.items():
                logger.info(f"  {name:30s}: {grad_norm:.2e}")
            
            # Check that gradients flow
            params_with_grad = len([p for p in adapter.parameters() if p.grad is not None])
            total_params = sum(1 for p in adapter.parameters())
            
            assert params_with_grad > 0, "No gradients computed"
            logger.info(f"✓ Gradients computed for {params_with_grad}/{total_params} parameters")
            results['params_with_grad'] = params_with_grad
            
            # Check for reasonable gradient magnitudes (not exploding/vanishing)
            grad_norms = [p.grad.norm().item() for p in adapter.parameters() if p.grad is not None]
            min_grad = min(grad_norms)
            max_grad = max(grad_norms)
            
            logger.info(f"✓ Gradient norm range: [{min_grad:.2e}, {max_grad:.2e}]")
            
            # Warn if gradients are very small (vanishing) or very large (exploding)
            if max_grad > 1e2:
                logger.warning("⚠ Warning: Maximum gradient norm is very large (potential explosion)")
                results['gradient_warning'] = 'exploding'
            elif min_grad < 1e-6 and max_grad > 1e-6:
                logger.warning("⚠ Warning: Some gradients are very small (potential vanishing)")
                results['gradient_warning'] = 'vanishing'
            else:
                logger.info("✓ Gradient magnitudes appear reasonable")
                results['gradient_warning'] = None
            
            results['status'] = 'PASSED'
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
        
        return results
    
    def test_numerical_stability(self) -> Dict[str, float]:
        """
        Test numerical stability with extreme inputs.
        
        Verifies:
        - Handles very small values (< 1e-6)
        - Handles very large values (> 1e3)
        - No NaN/Inf in outputs
        - Stable loss computation
        
        Returns:
            Dictionary with test results
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 5: Numerical Stability with Extreme Inputs")
        logger.info("="*70)
        
        try:
            from biomedclip_finetuning.frequency_adapter.svd_adapter import SVD_Frequency_Adapter
        except ImportError as e:
            logger.error(f"Could not import SVD_Frequency_Adapter: {e}")
            return {'status': 'FAILED', 'error': str(e)}
        
        results = {}
        tests_passed = 0
        
        try:
            adapter = SVD_Frequency_Adapter(d_model=768, rank_k=64).to(self.device)
            adapter.eval()
            
            test_cases = [
                ("Normal inputs", torch.randn(2, 197, 768, device=self.device)),
                ("Very small inputs", torch.randn(2, 197, 768, device=self.device) * 1e-6),
                ("Very large inputs", torch.randn(2, 197, 768, device=self.device) * 1e3),
                ("Mixed scale inputs", torch.cat([
                    torch.randn(2, 100, 768, device=self.device) * 1e-3,
                    torch.randn(2, 97, 768, device=self.device) * 1e3
                ], dim=1)),
            ]
            
            for test_name, x in test_cases:
                with torch.no_grad():
                    try:
                        y = adapter(x)
                        
                        # Check output
                        has_nan = torch.isnan(y).any().item()
                        has_inf = torch.isinf(y).any().item()
                        
                        if has_nan or has_inf:
                            logger.error(f"✗ {test_name}: Output contains NaN/Inf")
                            continue
                        
                        output_norm = torch.norm(y).item()
                        logger.info(f"✓ {test_name:25s} -> output norm: {output_norm:.2e}")
                        tests_passed += 1
                    
                    except Exception as e:
                        logger.error(f"✗ {test_name}: {e}")
                        continue
            
            results['tests_passed'] = tests_passed
            results['tests_total'] = len(test_cases)
            results['status'] = 'PASSED' if tests_passed == len(test_cases) else 'PARTIAL'
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
        
        return results
    
    def run_all_tests(self) -> Dict:
        """
        Run all ablation tests.
        
        Returns:
            Dictionary with all test results
        """
        logger.info("\n\n")
        logger.info("█" * 70)
        logger.info("█" + " " * 68 + "█")
        logger.info("█" + "  FREQUENCY ADAPTER V2 - ABLATION TESTING SUITE".center(68) + "█")
        logger.info("█" + " " * 68 + "█")
        logger.info("█" * 70)
        
        all_results = {
            'test_1_laplacian_pyramid': self.test_laplacian_pyramid(),
            'test_2_gamma_clamping': self.test_gamma_clamping(),
            'test_3_multi_stream_fusion': self.test_multi_stream_fusion(),
            'test_4_backward_pass': self.test_backward_pass(),
            'test_5_numerical_stability': self.test_numerical_stability(),
        }
        
        # Summary
        logger.info("\n\n")
        logger.info("█" * 70)
        logger.info("█" + " " * 68 + "█")
        logger.info("█" + "  TEST SUMMARY".center(68) + "█")
        logger.info("█" + " " * 68 + "█")
        logger.info("█" * 70)
        
        passed = sum(1 for r in all_results.values() if r.get('status') == 'PASSED')
        total = len(all_results)
        
        for test_name, result in all_results.items():
            status = result.get('status', 'UNKNOWN')
            symbol = "✓" if status == "PASSED" else "✗" if status == "FAILED" else "◐"
            logger.info(f"{symbol} {test_name}: {status}")
        
        logger.info("\n" + "─" * 70)
        logger.info(f"Overall: {passed}/{total} tests passed")
        logger.info("─" * 70 + "\n")
        
        return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Frequency Adapter V2 - Ablation Testing')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run tests on')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'laplacian', 'gamma', 'stream', 'backward', 'stability'],
                       help='Which test to run')
    args = parser.parse_args()
    
    tester = AblationTester(device=args.device)
    
    if args.test == 'all':
        results = tester.run_all_tests()
    elif args.test == 'laplacian':
        results = tester.test_laplacian_pyramid()
    elif args.test == 'gamma':
        results = tester.test_gamma_clamping()
    elif args.test == 'stream':
        results = tester.test_multi_stream_fusion()
    elif args.test == 'backward':
        results = tester.test_backward_pass()
    elif args.test == 'stability':
        results = tester.test_numerical_stability()
    
    print("\n" + "="*70)
    if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
        # Multiple tests
        print(f"All tests completed. See summary above.")
    else:
        # Single test
        print(f"Test result: {results.get('status', 'UNKNOWN')}")
    print("="*70)
