#!/usr/bin/env python3
"""
测试 DiffSTG 数据路径是否正确
"""

import os
import sys

# 模拟 DiffSTG train.py 中的路径计算
ws = "/home/xiaoxiao/TrafficFM-main/methods/baselines/DiffSTG"

baseline_root = ws
trafficfm_root = os.path.dirname(os.path.dirname(os.path.dirname(baseline_root)))

print("DiffSTG 路径计算测试:")
print("=" * 80)
print(f"baseline_root (ws): {baseline_root}")
print(f"trafficfm_root: {trafficfm_root}")
print()

# 测试数据路径
data_path = os.path.join(trafficfm_root, 'data', 'processed', 'DiffSTG') + '/'
print(f"data_path: {data_path}")
print()

# 测试各个数据集
datasets = ["metr-la", "pems-bay", "PEMS08", "AIR_GZ"]
for dataset in datasets:
    dataset_path = os.path.join(data_path, dataset)
    exists = os.path.exists(dataset_path)
    status = "✓" if exists else "✗"
    print(f"{status} {dataset} 数据路径: {dataset_path}")

    if exists:
        # 检查数据文件
        flow_file = os.path.join(dataset_path, "flow.npy")
        adj_file = os.path.join(dataset_path, "adj.npy")

        flow_exists = os.path.exists(flow_file)
        adj_exists = os.path.exists(adj_file)

        print(f"  {'✓' if flow_exists else '✗'} flow.npy")
        print(f"  {'✓' if adj_exists else '✗'} adj.npy")

print()
print("=" * 80)
print("路径验证完成！")
