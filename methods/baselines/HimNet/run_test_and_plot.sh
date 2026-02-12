#!/bin/bash
# HimNet Test and Plot Runner Script
# Quick and easy testing with STGCN-style interface

echo "=================================================="
echo "HimNet Model Testing and Visualization"
echo "STGCN-Style Interface with CRPS and WD Metrics"
echo "=================================================="
echo ""

# 默认参数
DATASET=${1:-"PEMSBAY"}
DEVICE=${2:-"cuda:0"}

# 检查是否在正确的目录
if [ ! -f "test_and_plot.py" ]; then
    echo "Error: test_and_plot.py not found!"
    echo "Please run this script from the HimNet-main directory."
    exit 1
fi

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Device:  $DEVICE"
echo ""

# 检查数据集参数
if [ "$DATASET" != "METRLA" ] && [ "$DATASET" != "PEMSBAY" ]; then
    echo "Error: Invalid dataset '$DATASET'"
    echo "Please use METRLA or PEMSBAY"
    echo ""
    echo "Usage:"
    echo "  ./run_test_and_plot.sh [DATASET] [DEVICE]"
    echo ""
    echo "Examples:"
    echo "  ./run_test_and_plot.sh METRLA cuda:0"
    echo "  ./run_test_and_plot.sh PEMSBAY cuda:1"
    echo "  ./run_test_and_plot.sh PEMSBAY cpu"
    exit 1
fi

echo "Starting test..."
echo ""

# 运行测试脚本
python3 test_and_plot.py --dataset $DATASET --device $DEVICE

echo ""
echo "=================================================="
echo "Testing completed!"
echo "=================================================="
