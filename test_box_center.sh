#!/bin/bash

# Script to test SAM with box+center point prompts (Heuristic B)
# This uses the improved LG-SR masks as input prompts

echo "=========================================="
echo "Testing Heuristic B: Box + Center Point"
echo "=========================================="

# Use improved LG-SR masks as input
MASK_INPUT="/home/long/projects/MedCLIP-SAMv3/results/LG_SR_improved"
IMAGE_INPUT="/home/long/projects/MedCLIP-SAMv3/data/brain_tumors/test_images"
OUTPUT_DIR="/home/long/projects/MedCLIP-SAMv3/results/sam_box_center"
GT_PATH="/home/long/projects/MedCLIP-SAMv3/data/brain_tumors/test_masks"

echo ""
echo "Step 1: Running SAM with box_center prompts..."
python segment-anything/prompt_sam.py \
    --input ${IMAGE_INPUT} \
    --mask-input ${MASK_INPUT} \
    --output ${OUTPUT_DIR} \
    --model-type vit_h \
    --checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
    --prompts box_center

echo ""
echo "Step 2: Evaluating results..."
python evaluation/eval.py \
    --gt_path ${GT_PATH} \
    --seg_path ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
