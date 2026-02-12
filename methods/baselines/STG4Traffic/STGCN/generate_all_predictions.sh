#!/bin/bash

# Script to generate STGCN predictions for both PEMSBAY and METRLA datasets
# Output format: (seq_length, num_samples, num_nodes)

echo "=========================================="
echo "STGCN Prediction NPY Generator"
echo "=========================================="
echo ""

# Set output directory
OUTPUT_DIR="./predictions_npy"
mkdir -p "$OUTPUT_DIR"

# Device to use (change if needed)
DEVICE="cuda:0"

echo "Python: $(which python)"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""

# Generate predictions for PEMSBAY
echo "=========================================="
echo "Generating predictions for PEMSBAY..."
echo "=========================================="
python generate_predictions_npy.py \
    --dataset PEMSBAY \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✓ PEMSBAY predictions generated successfully"
else
    echo "✗ Failed to generate PEMSBAY predictions"
fi

echo ""

# Generate predictions for METRLA
echo "=========================================="
echo "Generating predictions for METRLA..."
echo "=========================================="
python generate_predictions_npy.py \
    --dataset METRLA \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "✓ METRLA predictions generated successfully"
else
    echo "✗ Failed to generate METRLA predictions"
fi

echo ""
echo "=========================================="
echo "All predictions generated!"
echo "=========================================="
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.npy
echo ""
echo "Done!"
