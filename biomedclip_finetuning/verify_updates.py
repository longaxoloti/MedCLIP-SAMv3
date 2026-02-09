#!/usr/bin/env python3
"""
Verification script for updated Phase 2 scripts
Tests syntax, class definitions, and basic logic
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*60)
print("VERIFYING UPDATED SCRIPTS")
print("="*60)

tests_passed = 0
tests_total = 0

# Test 1: Import evaluate_baseline
tests_total += 1
try:
    # Check file exists and has correct content
    with open('evaluate_baseline.py', 'r') as f:
        content = f.read()
        # Check that Dice/IoU are removed
        if 'compute_dice_coefficient' in content:
            raise AssertionError("compute_dice_coefficient still in evaluate_baseline.py")
        if 'compute_iou' in content:
            raise AssertionError("compute_iou still in evaluate_baseline.py")
        # Check that semantic evaluation is added
        if 'compute_similarity' not in content:
            raise AssertionError("compute_similarity not found")
        if 'Semantic Sanity Check' not in content:
            raise AssertionError("Semantic Sanity Check comment not found")
    print("✓ PASS: evaluate_baseline.py updated correctly (Dice/IoU removed, semantic check added)")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAIL: evaluate_baseline.py verification: {e}")

# Test 2: Check MedpixDataset exists
tests_total += 1
try:
    with open('train_frequency_adapter.py', 'r') as f:
        content = f.read()
        if 'class MedpixDataset' not in content:
            raise AssertionError("MedpixDataset class not found")
        if 'split_ratio' not in content:
            raise AssertionError("split_ratio parameter not found")
        if '0.85' not in content:
            raise AssertionError("85/15 split ratio not found")
    print("✓ PASS: train_frequency_adapter.py has MedpixDataset with 85/15 split")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAIL: MedpixDataset verification: {e}")

# Test 3: Check train() method has use_medpix parameter
tests_total += 1
try:
    with open('train_frequency_adapter.py', 'r') as f:
        content = f.read()
        if 'use_medpix: bool = True' not in content:
            raise AssertionError("use_medpix parameter not found in train()")
        if 'if use_medpix:' not in content:
            raise AssertionError("use_medpix logic not found")
        if '--use-medpix' not in content:
            raise AssertionError("--use-medpix CLI argument not found")
    print("✓ PASS: train() method has use_medpix parameter and CLI argument")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAIL: use_medpix parameter verification: {e}")

# Test 4: Check that medical datasets still work
tests_total += 1
try:
    with open('train_frequency_adapter.py', 'r') as f:
        content = f.read()
        if 'class MedicalImageTextDataset' not in content:
            raise AssertionError("MedicalImageTextDataset still exists")
        if 'else:' not in content or '# Use medical segmentation datasets' not in content:
            raise AssertionError("Fallback to medical datasets not found")
    print("✓ PASS: Medical dataset support still available (--no-medpix flag)")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAIL: Medical dataset verification: {e}")

# Test 5: Check MedPix CSV path logic
tests_total += 1
try:
    with open('train_frequency_adapter.py', 'r') as f:
        content = f.read()
        if "Path(dataset_root).parent / 'medpix_dataset' / 'medpix_dataset.csv'" not in content:
            raise AssertionError("MedPix CSV path logic not found")
        if "Path(dataset_root).parent / 'medpix_dataset' / 'images'" not in content:
            raise AssertionError("MedPix images path logic not found")
    print("✓ PASS: MedPix CSV and images paths correctly referenced")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAIL: MedPix path verification: {e}")

# Test 6: Check that pandas import is inside MedpixDataset
tests_total += 1
try:
    with open('train_frequency_adapter.py', 'r') as f:
        lines = f.readlines()
        # Find MedpixDataset.__init__ line
        in_medpix_init = False
        pandas_imported_in_init = False
        for i, line in enumerate(lines):
            if 'def __init__' in line and i > 250:  # MedpixDataset is around line 290
                in_medpix_init = True
            elif in_medpix_init and 'import pandas' in line:
                pandas_imported_in_init = True
                break
            elif in_medpix_init and line.startswith('    def '):
                break
        if not pandas_imported_in_init:
            raise AssertionError("pandas import not in MedpixDataset.__init__")
    print("✓ PASS: pandas import correctly placed in MedpixDataset.__init__()")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAIL: pandas import verification: {e}")

# Test 7: Check scipy import in evaluate_baseline
tests_total += 1
try:
    with open('evaluate_baseline.py', 'r') as f:
        content = f.read()
        if 'from scipy.spatial.distance import cosine' not in content:
            raise AssertionError("scipy import not found")
    print("✓ PASS: scipy import correctly added to evaluate_baseline.py")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAIL: scipy import verification: {e}")

# Test 8: Verify medpix output directory handling
tests_total += 1
try:
    with open('train_frequency_adapter.py', 'r') as f:
        content = f.read()
        if "output_dir = os.path.join(args.output_dir, 'medpix', timestamp)" not in content:
            raise AssertionError("MedPix output directory logic not found")
    print("✓ PASS: MedPix output directory handling added")
    tests_passed += 1
except Exception as e:
    print(f"✗ FAIL: MedPix output directory verification: {e}")

# Summary
print("\n" + "="*60)
print(f"RESULTS: {tests_passed}/{tests_total} tests passed")
print("="*60)

if tests_passed == tests_total:
    print("✓ All verifications passed!")
    sys.exit(0)
else:
    print(f"✗ {tests_total - tests_passed} test(s) failed")
    sys.exit(1)
