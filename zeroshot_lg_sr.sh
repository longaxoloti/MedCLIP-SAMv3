#!/bin/bash
# Zero-shot segmentation pipeline with Laplacian-Guided Saliency Refinement (LG-SR)
# This script runs the full MedCLIP-SAMv2 pipeline with LG-SR post-processing enabled
# 
# Usage:
#   bash zeroshot_lg_sr.sh <dataset_name> <data_path> <output_dir>
#
# Example:
#   bash zeroshot_lg_sr.sh brain_tumors ./data/brain_images ./results_with_lg_sr

set -e

# Configuration
DATASET_NAME=${1:-"brain_tumors"}
DATA_PATH=${2:-"./data"}
OUTPUT_DIR=${3:-"./results_lg_sr"}

# LG-SR Parameters
LG_SR_ALPHA=0.5              # Fusion gain factor (0.5 = good balance)
LG_SR_EDGE_THRESHOLD=0.1     # Threshold for weak edges
USE_WATERSHED=0              # 0 = morphological fallback, 1 = watershed fallback
VERBOSE=1                     # 1 = print debug info, 0 = silent

# Directories
SALIENCY_OUTPUT="${OUTPUT_DIR}/saliency_maps"
COARSE_OUTPUT="${OUTPUT_DIR}/coarse_outputs"
FINAL_OUTPUT="${OUTPUT_DIR}/sam_outputs"
LOGS_DIR="${OUTPUT_DIR}/logs"

# Create output directories
mkdir -p "$SALIENCY_OUTPUT" "$COARSE_OUTPUT" "$FINAL_OUTPUT" "$LOGS_DIR"

echo "======================================================================"
echo "MedCLIP-SAMv2 Zero-Shot Segmentation with Laplacian-Guided SR (LG-SR)"
echo "======================================================================"
echo "Dataset: $DATASET_NAME"
echo "Data Path: $DATA_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "LG-SR Configuration:"
echo "  Alpha (fusion gain): $LG_SR_ALPHA"
echo "  Edge threshold: $LG_SR_EDGE_THRESHOLD"
echo "  Fallback method: $([ $USE_WATERSHED -eq 1 ] && echo 'Watershed' || echo 'Morphological')"
echo "======================================================================"
echo ""

# Step 1: Generate Saliency Maps (BiomedCLIP + M2IB)
echo "[1/3] Generating saliency maps using BiomedCLIP..."
python saliency_maps/generate_saliency_maps.py \
    --images_dir "$DATA_PATH/images" \
    --masks_dir "$DATA_PATH/masks" \
    --output_dir "$SALIENCY_OUTPUT" \
    --dataset_name "$DATASET_NAME" \
    2>&1 | tee "$LOGS_DIR/step1_saliency.log"

echo "✓ Saliency maps generated"
echo ""

# Step 2: Post-processing with K-means + LG-SR
echo "[2/3] Post-processing with K-means + Laplacian-Guided Saliency Refinement..."
python postprocessing/postprocess_saliency_maps.py \
    --postprocess kmeans \
    --input-path "$DATA_PATH/images" \
    --sal-path "$SALIENCY_OUTPUT" \
    --output-path "$COARSE_OUTPUT" \
    --num-contours 1 \
    --use-lg-sr \
    --lg-sr-alpha "$LG_SR_ALPHA" \
    --lg-sr-edge-threshold "$LG_SR_EDGE_THRESHOLD" \
    $([ $USE_WATERSHED -eq 1 ] && echo "--lg-sr-use-watershed" || echo "") \
    $([ $VERBOSE -eq 1 ] && echo "--verbose" || echo "") \
    2>&1 | tee "$LOGS_DIR/step2_postprocessing.log"

echo "✓ Post-processing with LG-SR completed"
echo ""

# Step 3: SAM Segmentation
echo "[3/3] Running SAM segmentation..."
python segment-anything/prompt_sam.py \
    --input_folder "$DATA_PATH/images" \
    --masks_folder "$COARSE_OUTPUT" \
    --output_folder "$FINAL_OUTPUT" \
    --model_type "vit_b" \
    --prompts "boxes" \
    2>&1 | tee "$LOGS_DIR/step3_sam.log"

echo "✓ SAM segmentation completed"
echo ""

echo "======================================================================"
echo "Pipeline completed successfully!"
echo "Output saved to: $OUTPUT_DIR"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Evaluate results: python evaluation/eval.py --pred_dir $FINAL_OUTPUT --gt_dir $DATA_PATH/masks"
echo "  2. Visualize outputs: Check $FINAL_OUTPUT for segmentation masks"
echo "  3. Compare with baseline: Run zeroshot.sh to compare with standard pipeline"
echo ""
