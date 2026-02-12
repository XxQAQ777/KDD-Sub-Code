#!/usr/bin/env python3
"""
验证 TrafficFM-main 集成的路径配置
"""

import os
import sys

def check_path(path, description):
    """检查路径是否存在"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("=" * 80)
    print("TrafficFM-main 集成验证")
    print("=" * 80)

    # 获取 TrafficFM-main 根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trafficfm_root = script_dir

    print(f"\nTrafficFM-main 根目录: {trafficfm_root}\n")

    # 检查目录结构
    print("检查目录结构:")
    print("-" * 80)

    all_ok = True

    # 检查主要目录
    all_ok &= check_path(os.path.join(trafficfm_root, "data"), "data/")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/raw"), "data/raw/")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed"), "data/processed/")
    all_ok &= check_path(os.path.join(trafficfm_root, "methods"), "methods/")
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/TrafficFM"), "methods/TrafficFM/")
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/baselines"), "methods/baselines/")
    all_ok &= check_path(os.path.join(trafficfm_root, "scripts"), "scripts/")
    all_ok &= check_path(os.path.join(trafficfm_root, "configs"), "configs/")

    print()

    # 检查原始数据
    print("检查原始数据:")
    print("-" * 80)
    all_ok &= check_path(os.path.join(trafficfm_root, "data/raw/metr-la.h5"), "METR-LA 原始数据")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/raw/pems-bay.h5"), "PEMS-BAY 原始数据")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/raw/sensor_graph"), "传感器图结构")

    print()

    # 检查 TrafficFM
    print("检查 TrafficFM:")
    print("-" * 80)
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/TrafficFM/train_metr.py"), "train_metr.py")
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/TrafficFM/train_pems.py"), "train_pems.py")
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/TrafficFM/engine.py"), "engine.py")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/TrafficFM"), "TrafficFM 数据目录")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/TrafficFM/METR-LA-144-3feat"), "METR-LA 数据")

    print()

    # 检查 HimNet
    print("检查 HimNet:")
    print("-" * 80)
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/baselines/HimNet"), "HimNet 代码目录")
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/baselines/HimNet/scripts/train.py"), "train.py")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/HimNet"), "HimNet 数据目录")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/HimNet/METRLA"), "METRLA 数据")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/HimNet/PEMSBAY"), "PEMSBAY 数据")

    # 确认 HimNet 的 data 目录已删除
    himnet_data = os.path.join(trafficfm_root, "methods/baselines/HimNet/data")
    if os.path.exists(himnet_data):
        print(f"✗ HimNet data/ 目录应该被删除: {himnet_data}")
        all_ok = False
    else:
        print(f"✓ HimNet data/ 目录已正确删除")

    print()

    # 检查 DiffSTG
    print("检查 DiffSTG:")
    print("-" * 80)
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/baselines/DiffSTG"), "DiffSTG 代码目录")
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/baselines/DiffSTG/train.py"), "train.py")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/DiffSTG"), "DiffSTG 数据目录")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/DiffSTG/metr-la"), "metr-la 数据")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/DiffSTG/pems-bay"), "pems-bay 数据")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/DiffSTG/metr-la/flow.npy"), "flow.npy")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/DiffSTG/metr-la/adj.npy"), "adj.npy")

    # 确认 DiffSTG 的 data 目录已删除
    diffstg_data = os.path.join(trafficfm_root, "methods/baselines/DiffSTG/data")
    if os.path.exists(diffstg_data):
        print(f"✗ DiffSTG data/ 目录应该被删除: {diffstg_data}")
        all_ok = False
    else:
        print(f"✓ DiffSTG data/ 目录已正确删除")

    print()

    # 检查 STG4Traffic
    print("检查 STG4Traffic (DCRNN, DGCRN, GWNET, STGCN):")
    print("-" * 80)
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/baselines/STG4Traffic"), "STG4Traffic 目录")
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/baselines/STG4Traffic/path_config.py"), "path_config.py")
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/baselines/STG4Traffic/lib"), "共享库 lib/")
    all_ok &= check_path(os.path.join(trafficfm_root, "methods/baselines/STG4Traffic/model"), "模型定义 model/")

    # 检查四个模型
    for model in ['DCRNN', 'DGCRN', 'GWNET', 'STGCN']:
        all_ok &= check_path(os.path.join(trafficfm_root, f"methods/baselines/STG4Traffic/{model}"), f"{model} 目录")

    # 检查数据
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/STG4Traffic"), "STG4Traffic 数据目录")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/STG4Traffic/METR-LA/processed"), "METR-LA 数据")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/STG4Traffic/PEMS-BAY/processed"), "PEMS-BAY 数据")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/STG4Traffic/METR-LA/processed/train.npz"), "train.npz")
    all_ok &= check_path(os.path.join(trafficfm_root, "data/processed/STG4Traffic/METR-LA/processed/adj_mx.pkl"), "adj_mx.pkl")

    print()

    # 检查文档
    print("检查文档:")
    print("-" * 80)
    all_ok &= check_path(os.path.join(trafficfm_root, "README.md"), "README.md")
    all_ok &= check_path(os.path.join(trafficfm_root, "BASELINE_INTEGRATION_PROTOCOL.md"), "集成协议")
    all_ok &= check_path(os.path.join(trafficfm_root, "INTEGRATION_LOG.md"), "集成记录")

    print()
    print("=" * 80)

    if all_ok:
        print("✓ 所有检查通过！集成成功。")
        return 0
    else:
        print("✗ 部分检查失败，请检查上述错误。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
