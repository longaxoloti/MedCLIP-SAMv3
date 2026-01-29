#!/bin/bash

# A/B Testing Script: Compare K-means vs K-means + LG-SR
#
# Usage: bash zeroshot_ab_test.sh <dataset_path>
# This script runs the pipeline twice:
#   1. Standard K-means post-processing
#   2. K-means + LG-SR refinement
# Allows for side-by-side comparison of results

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: bash zeroshot_ab_test.sh <dataset_path>"
    exit 1
fi

echo "=========================================="
echo "MedCLIP-SAMv2 A/B Testing: K-means vs LG-SR"
echo "=========================================="
echo "Dataset: $DATASET"
echo ""

# Step 1: Generate saliency maps (same for both)
echo "[Common Step] Generating saliency maps with BiomedCLIP..."
python saliency_maps/generate_saliency_maps.py \
--input-path ${DATASET}/images \
--output-path saliency_map_outputs/${DATASET}/masks \
--val-path ${DATASET}/val_images \
--model-name BiomedCLIP \
--finetuned \
--hyper-opt \
--val-path ${DATASET}/val_images

if [ $? -ne 0 ]; then
    echo "Error: Saliency map generation failed!"
    exit 1
fi

echo "✓ Saliency maps generated"
echo ""

# Test A: Standard K-means only
echo "=========================================="
echo "[Test A] Standard K-means Post-processing"
echo "=========================================="
python postprocessing/postprocess_saliency_maps.py \
--input-path ${DATASET}/images \
--output-path coarse_outputs/${DATASET}/masks_kmeans \
--sal-path saliency_map_outputs/${DATASET}/masks \
--postprocess kmeans \
--filter

if [ $? -ne 0 ]; then
    echo "Error: K-means post-processing failed!"
    exit 1
fi

echo "✓ K-means masks saved to: coarse_outputs/${DATASET}/masks_kmeans/"
echo ""

echo "Running SAM with K-means prompts..."
python segment-anything/prompt_sam.py \
--input ${DATASET}/images \
--mask-input coarse_outputs/${DATASET}/masks_kmeans \
--output sam_outputs/${DATASET}/masks_kmeans \
--model-type vit_h \
--checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
--prompts boxes

if [ $? -ne 0 ]; then
    echo "Error: SAM with K-means failed!"
    exit 1
fi

echo "✓ K-means SAM results saved to: sam_outputs/${DATASET}/masks_kmeans/"
echo ""

# Test B: K-means + LG-SR
echo "=========================================="
echo "[Test B] K-means + LG-SR Refinement"
echo "=========================================="
python postprocessing/postprocess_saliency_maps.py \
--input-path ${DATASET}/images \
--output-path coarse_outputs/${DATASET}/masks_lg_sr \
--sal-path saliency_map_outputs/${DATASET}/masks \
--postprocess kmeans \
--filter \
--use-lg-sr \
--lg-sr-alpha 0.5 \
--lg-sr-fallback \
--lg-sr-edge-threshold 0.1 \
--verbose

if [ $? -ne 0 ]; then
    echo "Error: LG-SR post-processing failed!"
    exit 1
fi

echo "✓ LG-SR masks saved to: coarse_outputs/${DATASET}/masks_lg_sr/"
echo ""

echo "Running SAM with LG-SR prompts..."
python segment-anything/prompt_sam.py \
--input ${DATASET}/images \
--mask-input coarse_outputs/${DATASET}/masks_lg_sr \
--output sam_outputs/${DATASET}/masks_lg_sr \
--model-type vit_h \
--checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
--prompts boxes

if [ $? -ne 0 ]; then
    echo "Error: SAM with LG-SR failed!"
    exit 1
fi

echo "✓ LG-SR SAM results saved to: sam_outputs/${DATASET}/masks_lg_sr/"
echo ""

echo "=========================================="
echo "✓ A/B Testing Complete!"
echo "=========================================="
echo ""
echo "Results Location:"
echo "  [Test A - K-means only]"
echo "    - Masks: coarse_outputs/${DATASET}/masks_kmeans/"
echo "    - Segmentation: sam_outputs/${DATASET}/masks_kmeans/"
echo ""
echo "  [Test B - K-means + LG-SR]"
echo "    - Masks: coarse_outputs/${DATASET}/masks_lg_sr/"
echo "    - Segmentation: sam_outputs/${DATASET}/masks_lg_sr/"
echo ""
echo "Next: Compare results between the two methods for evaluation"
echo "=========================================="
