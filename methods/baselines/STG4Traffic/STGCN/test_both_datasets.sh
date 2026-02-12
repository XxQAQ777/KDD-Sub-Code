#!/bin/bash

# STGCN 测试脚本 - 批量测试多个数据集
# 使用方法: bash test_both_datasets.sh

echo "========================================================================"
echo "STGCN Model Testing - Batch Script"
echo "========================================================================"
echo ""

# 设置默认参数
DEVICE="${1:-cuda:0}"  # 第一个参数是设备，默认cuda:0
BATCH_SIZE="${2:-64}"  # 第二个参数是batch size，默认64

echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# 测试 METR-LA
echo "========================================================================"
echo "Testing METR-LA Dataset"
echo "========================================================================"
python test_and_plot.py \
    --dataset METRLA \
    --device $DEVICE \
    --batch_size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ METR-LA testing completed successfully!"
    echo ""
else
    echo ""
    echo "✗ METR-LA testing failed!"
    echo ""
    exit 1
fi

# 测试 PEMS-BAY
echo "========================================================================"
echo "Testing PEMS-BAY Dataset"
echo "========================================================================"
python test_and_plot.py \
    --dataset PEMSBAY \
    --device $DEVICE \
    --batch_size $BATCH_SIZE

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ PEMS-BAY testing completed successfully!"
    echo ""
else
    echo ""
    echo "✗ PEMS-BAY testing failed!"
    echo ""
    exit 1
fi

echo "========================================================================"
echo "All tests completed!"
echo "========================================================================"
echo ""
echo "Results are saved in:"
ls -dt test_results_* | head -2
echo ""
