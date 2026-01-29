#!/usr/bin/env python
"""
A/B Testing Script: Compare Standard K-means vs LG-SR Refinement
================================================

This script runs both pipelines (with and without LG-SR) and compares:
- Binary mask quality (Dice, IoU)
- Bounding box tightness (area reduction)
- Edge quality metrics
- Computation time

Usage:
    python compare_lg_sr.py --input-path <images> --sal-path <saliency_maps> \
                           --gt-path <ground_truth> --output-dir <comparison_results>

Example:
    python compare_lg_sr.py \
        --input-path ./data/brain_images \
        --sal-path ./saliency_outputs \
        --gt-path ./data/brain_masks \
        --output-dir ./comparison_results
"""

import cv2
import numpy as np
import os
import argparse
import time
from tqdm import tqdm
from sklearn.cluster import KMeans
from laplacian_refinement import laplacian_guided_refine
import json


def dice_coefficient(pred, gt):
    """Calculate Dice Coefficient (F1-score)"""
    pred = (pred > 127).astype(np.uint8)
    gt = (gt > 127).astype(np.uint8)
    
    intersection = np.sum(pred & gt)
    if np.sum(pred) + np.sum(gt) == 0:
        return 1.0 if np.sum(pred) == np.sum(gt) else 0.0
    
    dice = 2 * intersection / (np.sum(pred) + np.sum(gt))
    return dice


def iou_coefficient(pred, gt):
    """Calculate Intersection over Union"""
    pred = (pred > 127).astype(np.uint8)
    gt = (gt > 127).astype(np.uint8)
    
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def get_bounding_box(mask):
    """Extract bounding box from mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    area = w * h
    return x, y, w, h, area


def postprocess_kmeans_baseline(saliency, original_image):
    """Standard K-means without LG-SR"""
    h, w = saliency.shape
    image = cv2.resize(saliency, (256, 256), interpolation=cv2.INTER_NEAREST)
    flat_image = image.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=2, random_state=10)
    labels = kmeans.fit_predict(flat_image)
    segmented = labels.reshape(256, 256)
    
    centroids = kmeans.cluster_centers_.flatten()
    background_cluster = np.argmin(centroids)
    
    segmented = np.where(segmented == background_cluster, 0, 1)
    segmented = cv2.resize(segmented, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return (segmented.astype(np.uint8) * 255)


def postprocess_kmeans_lgsr(saliency, original_image):
    """K-means with LG-SR refinement"""
    h, w = saliency.shape
    image = cv2.resize(saliency, (256, 256), interpolation=cv2.INTER_NEAREST)
    flat_image = image.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=2, random_state=10)
    labels = kmeans.fit_predict(flat_image)
    segmented = labels.reshape(256, 256)
    
    centroids = kmeans.cluster_centers_.flatten()
    background_cluster = np.argmin(centroids)
    
    segmented = np.where(segmented == background_cluster, 0, 1)
    segmented = cv2.resize(segmented, (w, h), interpolation=cv2.INTER_NEAREST)
    segmented = (segmented.astype(np.uint8) * 255)
    
    # Apply LG-SR
    refined = laplacian_guided_refine(
        segmented, 
        original_image, 
        alpha=0.5, 
        edge_threshold=0.1,
        use_watershed=False
    )
    
    return refined


def compare_pipelines(input_path, sal_path, gt_path, output_dir):
    """Compare baseline vs LG-SR on all images"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = sorted(os.listdir(sal_path))
    results = {
        'baseline': {'dice': [], 'iou': [], 'bbox_area': [], 'time': []},
        'lgsr': {'dice': [], 'iou': [], 'bbox_area': [], 'time': []},
        'improvement': {'dice': [], 'iou': [], 'bbox_reduction': []}
    }
    
    print(f"\n{'='*80}")
    print(f"{'A/B Testing: Baseline vs LG-SR':<80}")
    print(f"{'='*80}\n")
    print(f"{'Filename':<30} {'Dice (BL)':<12} {'Dice (LG-SR)':<12} {'IOU (BL)':<12} {'IOU (LG-SR)':<12} {'BBox Red %':<12}")
    print(f"{'-'*80}")
    
    comparison_data = []
    
    for file in tqdm(files, desc="Processing"):
        try:
            # Load inputs
            sal_img = cv2.imread(os.path.join(sal_path, file), 0) / 255.0
            orig_img = cv2.imread(os.path.join(input_path, file))
            gt_img = cv2.imread(os.path.join(gt_path, file), 0) if os.path.exists(os.path.join(gt_path, file)) else None
            
            if sal_img is None or orig_img is None:
                continue
            
            # Baseline: K-means without LG-SR
            t0 = time.time()
            baseline_mask = postprocess_kmeans_baseline(sal_img, orig_img)
            time_baseline = time.time() - t0
            
            # LG-SR: K-means with LG-SR
            t0 = time.time()
            lgsr_mask = postprocess_kmeans_lgsr(sal_img, orig_img)
            time_lgsr = time.time() - t0
            
            # Metrics
            if gt_img is not None:
                dice_bl = dice_coefficient(baseline_mask, gt_img)
                dice_lgsr = dice_coefficient(lgsr_mask, gt_img)
                iou_bl = iou_coefficient(baseline_mask, gt_img)
                iou_lgsr = iou_coefficient(lgsr_mask, gt_img)
                
                results['baseline']['dice'].append(dice_bl)
                results['baseline']['iou'].append(iou_bl)
                results['lgsr']['dice'].append(dice_lgsr)
                results['lgsr']['iou'].append(iou_lgsr)
                
                dice_improvement = (dice_lgsr - dice_bl) / (dice_bl + 1e-6) * 100
                iou_improvement = (iou_lgsr - iou_bl) / (iou_bl + 1e-6) * 100
                
                results['improvement']['dice'].append(dice_improvement)
                results['improvement']['iou'].append(iou_improvement)
            else:
                dice_bl = dice_lgsr = iou_bl = iou_lgsr = 0.0
            
            # Bounding box analysis
            bbox_bl = get_bounding_box(baseline_mask)
            bbox_lgsr = get_bounding_box(lgsr_mask)
            
            if bbox_bl and bbox_lgsr:
                area_bl = bbox_bl[4]
                area_lgsr = bbox_lgsr[4]
                bbox_reduction = (area_bl - area_lgsr) / (area_bl + 1e-6) * 100
                
                results['baseline']['bbox_area'].append(area_bl)
                results['lgsr']['bbox_area'].append(area_lgsr)
                results['improvement']['bbox_reduction'].append(bbox_reduction)
            else:
                bbox_reduction = 0.0
            
            results['baseline']['time'].append(time_baseline)
            results['lgsr']['time'].append(time_lgsr)
            
            # Print results
            print(f"{file:<30} {dice_bl:<12.4f} {dice_lgsr:<12.4f} {iou_bl:<12.4f} {iou_lgsr:<12.4f} {bbox_reduction:<12.2f}%")
            
            # Save comparison images
            comparison_img = np.hstack([
                cv2.cvtColor(baseline_mask, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(lgsr_mask, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR) if gt_img is not None else cv2.cvtColor(np.zeros_like(baseline_mask), cv2.COLOR_GRAY2BGR)
            ])
            
            comp_file = os.path.join(output_dir, f"comparison_{file}")
            cv2.imwrite(comp_file, comparison_img)
            
            comparison_data.append({
                'file': file,
                'baseline_dice': float(dice_bl),
                'lgsr_dice': float(dice_lgsr),
                'baseline_iou': float(iou_bl),
                'lgsr_iou': float(iou_lgsr),
                'bbox_reduction_percent': float(bbox_reduction)
            })
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"{'Summary Statistics':<80}")
    print(f"{'='*80}\n")
    
    metrics = {}
    for method in ['baseline', 'lgsr']:
        metrics[method] = {}
        if results[method]['dice']:
            metrics[method]['dice_mean'] = float(np.mean(results[method]['dice']))
            metrics[method]['dice_std'] = float(np.std(results[method]['dice']))
        if results[method]['iou']:
            metrics[method]['iou_mean'] = float(np.mean(results[method]['iou']))
            metrics[method]['iou_std'] = float(np.std(results[method]['iou']))
        if results[method]['bbox_area']:
            metrics[method]['bbox_area_mean'] = float(np.mean(results[method]['bbox_area']))
        if results[method]['time']:
            metrics[method]['time_mean'] = float(np.mean(results[method]['time']))
    
    improvements = {}
    if results['improvement']['dice']:
        improvements['dice_improvement_percent'] = float(np.mean(results['improvement']['dice']))
    if results['improvement']['iou']:
        improvements['iou_improvement_percent'] = float(np.mean(results['improvement']['iou']))
    if results['improvement']['bbox_reduction']:
        improvements['bbox_reduction_percent'] = float(np.mean(results['improvement']['bbox_reduction']))
    
    print(f"Metric{'':20}Baseline           LG-SR            Improvement")
    print(f"{'-'*80}")
    if metrics['baseline'].get('dice_mean'):
        print(f"Dice{'':<18}{metrics['baseline'].get('dice_mean', 0):<18.4f}{metrics['lgsr'].get('dice_mean', 0):<18.4f}{improvements.get('dice_improvement_percent', 0):>+.2f}%")
    if metrics['baseline'].get('iou_mean'):
        print(f"IOU{'':<19}{metrics['baseline'].get('iou_mean', 0):<18.4f}{metrics['lgsr'].get('iou_mean', 0):<18.4f}{improvements.get('iou_improvement_percent', 0):>+.2f}%")
    if metrics['baseline'].get('bbox_area_mean'):
        print(f"Bbox Area{'':<14}{metrics['baseline'].get('bbox_area_mean', 0):<18.1f}{metrics['lgsr'].get('bbox_area_mean', 0):<18.1f}{improvements.get('bbox_reduction_percent', 0):>-.2f}%")
    if metrics['baseline'].get('time_mean'):
        print(f"Time (ms){'':<15}{metrics['baseline'].get('time_mean', 0)*1000:<18.2f}{metrics['lgsr'].get('time_mean', 0)*1000:<18.2f}+{(metrics['lgsr'].get('time_mean', 0) - metrics['baseline'].get('time_mean', 0))*1000:>+.2f}ms")
    
    print(f"\n{'='*80}")
    print(f"Comparison results saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    # Save results to JSON
    results_json = {
        'summary': {
            'baseline': metrics.get('baseline', {}),
            'lgsr': metrics.get('lgsr', {}),
            'improvements': improvements
        },
        'detailed': comparison_data
    }
    
    with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Detailed results saved to: {os.path.join(output_dir, 'comparison_results.json')}")
    
    return results_json


def main():
    parser = argparse.ArgumentParser(description='A/B Testing: Baseline K-means vs LG-SR')
    parser.add_argument('--input-path', type=str, required=True, help='Path to original images')
    parser.add_argument('--sal-path', type=str, required=True, help='Path to saliency maps')
    parser.add_argument('--gt-path', type=str, help='Path to ground truth masks (optional)')
    parser.add_argument('--output-dir', type=str, default='./lg_sr_comparison', help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    compare_pipelines(args.input_path, args.sal_path, args.gt_path, args.output_dir)


if __name__ == '__main__':
    main()
