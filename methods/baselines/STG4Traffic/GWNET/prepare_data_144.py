#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
准备144时间步的数据
使用方法: python prepare_data_144.py
"""

import sys
sys.path.append('../')

import os
import numpy as np
from lib.generate_data import generate_data_h5


class Args:
    """数据生成参数"""
    def __init__(self):
        self.window = 144  # 输入窗口：144个时间步
        self.horizon = 144  # 预测窗口：144个时间步
        self.train_rate = 0.7  # 训练集比例
        self.val_rate = 0.1  # 验证集比例
        self.dataset = "METR-LA"  # 数据集名称


def check_data_exists(processed_dir):
    """检查数据文件是否存在"""
    required_files = ['train.npz', 'val.npz', 'test.npz']
    all_exist = True

    for file in required_files:
        filepath = os.path.join(processed_dir, file)
        if os.path.exists(filepath):
            # 检查数据形状
            data = np.load(filepath)
            x_shape = data['x'].shape
            y_shape = data['y'].shape
            print(f"✓ {file} exists - x: {x_shape}, y: {y_shape}")

            # 检查是否是144时间步
            if x_shape[1] == 144 and y_shape[1] == 144:
                print(f"  → Already 144 timesteps")
            else:
                print(f"  → WARNING: Not 144 timesteps, will regenerate")
                all_exist = False
        else:
            print(f"✗ {file} does not exist")
            all_exist = False

    return all_exist


def main():
    print("=" * 80)
    print("准备GWNET 144时间步数据")
    print("=" * 80)

    args = Args()

    # 设置路径
    if args.dataset == "METR-LA":
        h5_file = "../data/METR-LA/metr-la.h5"
        processed_dir = "../data/METR-LA/processed"
    elif args.dataset == "PEMS-BAY":
        h5_file = "../data/PEMS-BAY/pems-bay.h5"
        processed_dir = "../data/PEMS-BAY/processed"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"\n数据集: {args.dataset}")
    print(f"输入文件: {h5_file}")
    print(f"输出目录: {processed_dir}")
    print(f"输入窗口: {args.window} 时间步")
    print(f"预测窗口: {args.horizon} 时间步")
    print(f"训练/验证/测试比例: {args.train_rate}/{args.val_rate}/{1-args.train_rate-args.val_rate}")
    print("=" * 80)

    # 检查h5文件是否存在
    if not os.path.exists(h5_file):
        print(f"\n错误: 找不到数据文件 {h5_file}")
        print("请确保已下载METR-LA数据集")
        return

    # 创建输出目录
    os.makedirs(processed_dir, exist_ok=True)

    # 检查现有数据
    print("\n检查现有数据...")
    data_exists = check_data_exists(processed_dir)

    if data_exists:
        print("\n✓ 144时间步数据已存在，无需重新生成")
        response = input("是否要重新生成数据? (y/n): ")
        if response.lower() != 'y':
            print("跳过数据生成")
            return

    # 生成数据
    print("\n开始生成数据...")
    print("=" * 80)

    try:
        generate_data_h5(args, h5_file, processed_dir)
        print("=" * 80)
        print("\n✓ 数据生成成功!")

        # 验证生成的数据
        print("\n验证生成的数据:")
        check_data_exists(processed_dir)

    except Exception as e:
        print(f"\n✗ 数据生成失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("数据准备完成!")
    print("现在可以运行训练脚本: python train_gwnet_144.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
