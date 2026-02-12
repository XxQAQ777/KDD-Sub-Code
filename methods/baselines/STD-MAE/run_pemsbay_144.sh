#!/bin/bash

# Script to prepare PEMS-BAY dataset and run STD-MAE model with 144->144 prediction
# This script follows the same configuration as METR-LA (144->144)

set -e  # Exit on error

echo "=========================================="
echo "PEMS-BAY Dataset Preparation and Training"
echo "Configuration: 144 steps -> 144 steps"
echo "=========================================="
echo ""

# Change to project directory
cd /home/xiaoxiao/STD-MAE-main

# Step 1: Check if required data files exist
echo "Step 1: Checking data files..."
if [ ! -f "/home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/data/PEMS-BAY/pems-bay.h5" ]; then
    echo "ERROR: pems-bay.h5 not found!"
    exit 1
fi

if [ ! -f "/home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/data/PEMS-BAY/processed/adj_mx_bay.pkl" ]; then
    echo "ERROR: adj_mx_bay.pkl not found!"
    exit 1
fi

echo "✓ Data files found"
echo ""

# Step 2: Create output directory if it doesn't exist
echo "Step 2: Creating output directory..."
mkdir -p datasets/PEMS-BAY
echo "✓ Output directory ready"
echo ""

# Step 3: Generate training data
echo "Step 3: Generating PEMS-BAY training data (144->144)..."
python scripts/data_preparation/PEMS-BAY/generate_training_data_144.py

if [ $? -eq 0 ]; then
    echo "✓ Data generation completed successfully"
else
    echo "ERROR: Data generation failed!"
    exit 1
fi
echo ""

# Step 4: Verify generated files
echo "Step 4: Verifying generated files..."
if [ -f "datasets/PEMS-BAY/data_in144_out144.pkl" ] && \
   [ -f "datasets/PEMS-BAY/index_in144_out144.pkl" ] && \
   [ -f "datasets/PEMS-BAY/adj_mx.pkl" ]; then
    echo "✓ All required files generated:"
    echo "  - data_in144_out144.pkl"
    echo "  - index_in144_out144.pkl"
    echo "  - adj_mx.pkl"
    echo "  - scaler_in144_out144.pkl"
else
    echo "ERROR: Some required files are missing!"
    exit 1
fi
echo ""

# Step 5: Run the model
echo "Step 5: Starting STD-MAE training on PEMS-BAY..."
echo "Command: python stdmae/run.py --cfg='stdmae/STDMAE_PEMS-BAY_144.py'"
echo ""

python stdmae/run.py --cfg='stdmae/STDMAE_PEMS-BAY_144.py'

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
