#!/bin/bash

# Quick script to run STD-MAE on PEMS-BAY with 144->144 prediction
# Run this after data preparation is complete

cd /home/xiaoxiao/STD-MAE-main
python stdmae/run.py --cfg='stdmae/STDMAE_PEMS-BAY_144.py'
