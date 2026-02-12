#!/bin/bash

# Script to generate NPY prediction files for both METRLA and PEMSBAY datasets

echo "=========================================="
echo "Generating NPY Prediction Files"
echo "=========================================="
echo ""

# Create output directory
OUTPUT_DIR="npy_predictions"
mkdir -p $OUTPUT_DIR

# Generate predictions for METRLA
echo "=========================================="
echo "Processing METRLA dataset..."
echo "=========================================="
python generate_npy_predictions.py \
    --dataset METRLA \
    --output_dir ${OUTPUT_DIR}/METRLA \
    --device cuda:0

echo ""
echo "METRLA completed!"
echo ""

# Generate predictions for PEMSBAY
echo "=========================================="
echo "Processing PEMSBAY dataset..."
echo "=========================================="
python generate_npy_predictions.py \
    --dataset PEMSBAY \
    --output_dir ${OUTPUT_DIR}/PEMSBAY \
    --device cuda:0

echo ""
echo "PEMSBAY completed!"
echo ""

echo "=========================================="
echo "All NPY files generated successfully!"
echo "=========================================="
echo ""
echo "Output structure:"
echo "  ${OUTPUT_DIR}/"
echo "    ├── METRLA/"
echo "    │   ├── ground_truth.npy"
echo "    │   ├── himnet_predictions.npy"
echo "    │   └── himnet_metrla_predictions.npy"
echo "    └── PEMSBAY/"
echo "        ├── ground_truth.npy"
echo "        ├── himnet_predictions.npy"
echo "        └── himnet_pemsbay_predictions.npy"
echo ""
