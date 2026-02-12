#!/usr/bin/env python3
"""
测试 HimNet 数据路径是否正确
"""

import os
import sys

# 模拟 HimNet train.py 中的路径计算
script_path = "/home/xiaoxiao/TrafficFM-main/methods/baselines/HimNet/scripts/train.py"

baseline_root = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))
trafficfm_root = os.path.dirname(os.path.dirname(os.path.dirname(baseline_root)))

print("路径计算测试:")
print("=" * 80)
print(f"脚本路径: {script_path}")
print(f"baseline_root: {baseline_root}")
print(f"trafficfm_root: {trafficfm_root}")
print()

# 测试数据路径
for dataset in ["METRLA", "PEMSBAY"]:
    data_path = os.path.join(trafficfm_root, "data", "processed", "HimNet", dataset)
    exists = os.path.exists(data_path)
    status = "✓" if exists else "✗"
    print(f"{status} {dataset} 数据路径: {data_path}")

    if exists:
        # 检查数据文件
        for filename in ["train.npz", "val.npz", "test.npz"]:
            filepath = os.path.join(data_path, filename)
            file_exists = os.path.exists(filepath)
            file_status = "✓" if file_exists else "✗"
            print(f"  {file_status} {filename}")
