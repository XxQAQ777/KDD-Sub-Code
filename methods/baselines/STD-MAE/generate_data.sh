#!/bin/bash

# Install missing dependencies
echo "Installing missing dependencies..."
pip install pandas tables -q

# Generate METR-LA dataset with 144->144 configuration
echo "Generating METR-LA dataset (144->144)..."
cd /home/xiaoxiao/STD-MAE-main
python scripts/data_preparation/METR-LA/generate_training_data.py

echo "Data generation completed!"
