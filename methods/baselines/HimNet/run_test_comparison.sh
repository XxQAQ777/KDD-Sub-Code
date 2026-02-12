#!/bin/bash
# HimNet Test Comparison Runner Script

echo "=================================================="
echo "HimNet Model Testing and Visualization"
echo "=================================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "test_comparison.py" ]; then
    echo "Error: test_comparison.py not found!"
    echo "Please run this script from the HimNet-main directory."
    exit 1
fi

# 检查模型文件
echo "Checking model files..."
if [ ! -f "saved_models/HimNet-METRLA-2025-11-28-00-26-44_epoch24.pt" ]; then
    echo "Warning: METRLA model file not found!"
fi

if [ ! -f "saved_models/HimNet-PEMSBAY-2025-12-19-12-16-11_epoch13.pt" ]; then
    echo "Warning: PEMSBAY model file not found!"
fi

echo ""
echo "Starting test comparison..."
echo ""

# 运行测试脚本
# 默认使用GPU 0
GPU_NUM=${1:-0}

python3 test_comparison.py -g $GPU_NUM

echo ""
echo "=================================================="
echo "Testing completed!"
echo "=================================================="
