#!/usr/bin/env python
"""
Unit Test for Laplacian-Guided Saliency Refinement (LG-SR)
==========================================================

This test script validates the LG-SR module functions:
1. Laplacian edge detection
2. Adaptive fusion
3. Fallback methods
4. Full LG-SR pipeline

Usage:
    python test_lg_sr.py [--verbose]
"""

import sys
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postprocessing'))

from laplacian_refinement import (
    compute_laplacian_edges,
    adaptive_fusion,
    detect_edge_strength,
    morphological_fallback,
    watershed_fallback,
    laplacian_guided_refine
)


class TestLGSR:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.tests_passed = 0
        self.tests_failed = 0
        self.temp_dir = tempfile.mkdtemp()
    
    def log(self, msg):
        if self.verbose:
            print(msg)
    
    def assert_equal(self, actual, expected, name):
        if actual == expected:
            self.tests_passed += 1
            self.log(f"  ✓ {name}: PASSED")
            return True
        else:
            self.tests_failed += 1
            print(f"  ✗ {name}: FAILED (expected {expected}, got {actual})")
            return False
    
    def assert_in_range(self, value, min_val, max_val, name):
        if min_val <= value <= max_val:
            self.tests_passed += 1
            self.log(f"  ✓ {name}: PASSED ({value:.4f})")
            return True
        else:
            self.tests_failed += 1
            print(f"  ✗ {name}: FAILED ({value:.4f} not in [{min_val}, {max_val}])")
            return False
    
    def assert_shape(self, arr, expected_shape, name):
        if arr.shape == expected_shape:
            self.tests_passed += 1
            self.log(f"  ✓ {name}: PASSED {arr.shape}")
            return True
        else:
            self.tests_failed += 1
            print(f"  ✗ {name}: FAILED (expected {expected_shape}, got {arr.shape})")
            return False
    
    def test_laplacian_edge_detection(self):
        """Test Laplacian edge detection"""
        print("\n[TEST 1] Laplacian Edge Detection")
        print("-" * 60)
        
        # Create synthetic test image with clear edges
        img = np.zeros((100, 100), dtype=np.uint8)
        img[25:75, 25:75] = 255  # White square
        
        try:
            edge_map = compute_laplacian_edges(img, sigma=1.0)
            
            # Validate output
            self.assert_shape(edge_map, (100, 100), "Output shape")
            self.assert_in_range(edge_map.min(), 0.0, 1.0, "Min value in [0, 1]")
            self.assert_in_range(edge_map.max(), 0.0, 1.0, "Max value in [0, 1]")
            
            # Edges should be strong at boundaries (25-26, 74-75)
            edge_center = np.mean(edge_map[45:55, 45:55])
            edge_boundary = np.mean(edge_map[24:27, 24:27])
            
            if edge_boundary > edge_center:
                self.tests_passed += 1
                self.log(f"  ✓ Edge detection correctness: boundary ({edge_boundary:.4f}) > center ({edge_center:.4f})")
            else:
                self.tests_failed += 1
                print(f"  ✗ Edge detection correctness: boundary ({edge_boundary:.4f}) <= center ({edge_center:.4f})")
            
            print(f"  Edge map stats: min={edge_map.min():.4f}, max={edge_map.max():.4f}, mean={edge_map.mean():.4f}")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"  ✗ Exception: {str(e)}")
    
    def test_adaptive_fusion(self):
        """Test adaptive fusion formula"""
        print("\n[TEST 2] Adaptive Fusion")
        print("-" * 60)
        
        # Create synthetic masks
        binary_mask = np.ones((50, 50), dtype=np.uint8) * 128  # 50% foreground
        edge_map = np.linspace(0, 1, 50*50, dtype=np.float32).reshape(50, 50)
        
        try:
            refined = adaptive_fusion(binary_mask, edge_map, alpha=0.5)
            
            # Validate output
            self.assert_shape(refined, (50, 50), "Output shape")
            self.assert_in_range(refined.min(), 0.0, 1.0, "Min value in [0, 1]")
            self.assert_in_range(refined.max(), 0.0, 1.0, "Max value in [0, 1]")
            
            # Check that high-edge areas are reinforced
            reinforced_ratio = np.sum(refined > 0.5) / refined.size
            if reinforced_ratio > 0.3:  # Should be more than half reinforced
                self.tests_passed += 1
                self.log(f"  ✓ Reinforcement effect: {reinforced_ratio*100:.1f}% of pixels reinforced")
            else:
                self.tests_failed += 1
                print(f"  ✗ Reinforcement effect too weak: only {reinforced_ratio*100:.1f}% reinforced")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"  ✗ Exception: {str(e)}")
    
    def test_edge_strength_detection(self):
        """Test edge strength detection"""
        print("\n[TEST 3] Edge Strength Detection")
        print("-" * 60)
        
        # Weak edges
        weak_edge = np.ones((50, 50), dtype=np.float32) * 0.05
        strength_weak, is_weak_weak = detect_edge_strength(weak_edge, threshold=0.1)
        
        self.assert_equal(is_weak_weak, True, "Weak edge detection (should be True)")
        self.assert_in_range(strength_weak, 0.0, 0.1, f"Weak edge strength {strength_weak:.4f}")
        
        # Strong edges
        strong_edge = np.ones((50, 50), dtype=np.float32) * 0.5
        strength_strong, is_weak_strong = detect_edge_strength(strong_edge, threshold=0.1)
        
        self.assert_equal(is_weak_strong, False, "Strong edge detection (should be False)")
        self.assert_in_range(strength_strong, 0.4, 1.0, f"Strong edge strength {strength_strong:.4f}")
    
    def test_morphological_fallback(self):
        """Test morphological fallback method"""
        print("\n[TEST 4] Morphological Fallback")
        print("-" * 60)
        
        # Create noisy binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # Central region
        
        # Add noise
        noise = np.random.rand(100, 100) > 0.8
        mask[noise] = 255  # Random noise
        
        try:
            refined = morphological_fallback(mask, iterations=2, method='close')
            
            self.assert_shape(refined, (100, 100), "Output shape")
            self.assert_equal(refined.dtype, np.uint8, "Output dtype")
            
            # Refined should have less total area (noise removed)
            if np.sum(refined) < np.sum(mask):
                self.tests_passed += 1
                self.log(f"  ✓ Noise reduction: {np.sum(mask)} -> {np.sum(refined)} pixels")
            else:
                self.tests_failed += 1
                print(f"  ✗ No noise reduction: {np.sum(mask)} -> {np.sum(refined)} pixels")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"  ✗ Exception: {str(e)}")
    
    def test_full_pipeline(self):
        """Test full LG-SR pipeline"""
        print("\n[TEST 5] Full LG-SR Pipeline")
        print("-" * 60)
        
        try:
            # Create synthetic image with tumor-like feature
            image = np.zeros((150, 150, 3), dtype=np.uint8)
            # Add gradient background
            for i in range(150):
                image[i, :] = int(i / 150 * 100)
            # Add bright lesion
            image[60:90, 60:90] = 200
            
            # Create corresponding binary mask
            binary_mask = np.zeros((150, 150), dtype=np.uint8)
            binary_mask[55:95, 55:95] = 255  # Slightly larger region
            
            # Apply LG-SR
            refined_mask = laplacian_guided_refine(
                binary_mask,
                image,
                alpha=0.5,
                edge_threshold=0.1,
                use_watershed=False
            )
            
            # Validate output
            self.assert_shape(refined_mask, (150, 150), "Output shape")
            self.assert_equal(refined_mask.dtype, np.uint8, "Output dtype")
            
            # Refined should be more compact (smaller area due to edge-based refinement)
            area_before = np.sum(binary_mask > 127)
            area_after = np.sum(refined_mask > 127)
            
            if area_after < area_before:
                reduction = (area_before - area_after) / (area_before + 1e-6) * 100
                self.tests_passed += 1
                self.log(f"  ✓ Boundary refinement: {reduction:.1f}% area reduction")
            else:
                self.tests_failed += 1
                print(f"  ✗ No boundary refinement: area {area_before} -> {area_after}")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"  ✗ Exception: {str(e)}")
    
    def test_rgb_image(self):
        """Test LG-SR with RGB images"""
        print("\n[TEST 6] RGB Image Support")
        print("-" * 60)
        
        try:
            # Create RGB image
            image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
            image_rgb[30:70, 30:70, 0] = 200  # Red channel lesion
            
            binary_mask = np.zeros((100, 100), dtype=np.uint8)
            binary_mask[25:75, 25:75] = 255
            
            # Process
            refined = laplacian_guided_refine(
                binary_mask,
                image_rgb,
                alpha=0.5,
                edge_threshold=0.1
            )
            
            self.assert_shape(refined, (100, 100), "Output shape with RGB input")
            self.assert_equal(refined.dtype, np.uint8, "Output dtype with RGB input")
            
        except Exception as e:
            self.tests_failed += 1
            print(f"  ✗ Exception: {str(e)}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("LAPLACIAN-GUIDED SALIENCY REFINEMENT (LG-SR) TEST SUITE")
        print("="*60)
        
        self.test_laplacian_edge_detection()
        self.test_adaptive_fusion()
        self.test_edge_strength_detection()
        self.test_morphological_fallback()
        self.test_full_pipeline()
        self.test_rgb_image()
        
        # Summary
        total = self.tests_passed + self.tests_failed
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {total}")
        print(f"Passed: {self.tests_passed} ✓")
        print(f"Failed: {self.tests_failed} ✗")
        
        if self.tests_failed == 0:
            print("\n✓ All tests passed!")
            return 0
        else:
            print(f"\n✗ {self.tests_failed} test(s) failed")
            return 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test LG-SR module')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    tester = TestLGSR(verbose=args.verbose)
    exit_code = tester.run_all_tests()
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
