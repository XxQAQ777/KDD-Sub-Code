#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单测试：验证数据能否被正确加载
"""

import numpy as np
import sys
sys.path.append('/home/xiaoxiao/DiffSTG-main')

def test_data_files():
    """测试数据文件是否可以正确加载"""
    print("测试数据文件加载...")

    datasets = {
        'metr-la': {
            'flow_path': '/home/xiaoxiao/DiffSTG-main/data/dataset/metr-la/flow.npy',
            'adj_path': '/home/xiaoxiao/DiffSTG-main/data/dataset/metr-la/adj.npy',
            'expected_nodes': 207,
            'expected_features': 3
        },
        'pems-bay': {
            'flow_path': '/home/xiaoxiao/DiffSTG-main/data/dataset/pems-bay/flow.npy',
            'adj_path': '/home/xiaoxiao/DiffSTG-main/data/dataset/pems-bay/adj.npy',
            'expected_nodes': 325,
            'expected_features': 3
        }
    }

    for name, paths in datasets.items():
        print(f"\n{'='*60}")
        print(f"测试 {name}")
        print(f"{'='*60}")

        # 加载flow数据
        flow = np.load(paths['flow_path'])
        print(f"✓ Flow shape: {flow.shape}")

        T, V, D = flow.shape
        assert V == paths['expected_nodes'], f"节点数不匹配: {V} vs {paths['expected_nodes']}"
        assert D == paths['expected_features'], f"特征数不匹配: {D} vs {paths['expected_features']}"

        # 加载邻接矩阵
        adj = np.load(paths['adj_path'])
        print(f"✓ Adj shape: {adj.shape}")
        assert adj.shape == (V, V), f"邻接矩阵维度不匹配"

        # 检查特征
        print(f"\n特征检查:")
        print(f"  Feature 0 (flow): min={flow[:,:,0].min():.2f}, max={flow[:,:,0].max():.2f}")
        print(f"  Feature 1 (tod): min={flow[:,:,1].min():.0f}, max={flow[:,:,1].max():.0f}")
        print(f"  Feature 2 (dow): min={flow[:,:,2].min():.0f}, max={flow[:,:,2].max():.0f}")

        # 模拟数据划分
        train_size = int(T * 0.7)
        val_size = int(T * 0.1)

        print(f"\n数据划分:")
        print(f"  训练集: 0 - {train_size}")
        print(f"  验证集: {train_size} - {train_size + val_size}")
        print(f"  测试集: {train_size + val_size} - {T}")

        # 检查是否有足够的数据用于144步预测
        min_samples_needed = 144 * 2  # 历史144 + 预测144
        print(f"\n序列长度检查:")
        print(f"  最小需要: {min_samples_needed} 步")
        print(f"  训练集可用: {train_size} 步")
        print(f"  验证集可用: {val_size} 步")

        if train_size < min_samples_needed:
            print(f"  ⚠️  警告: 训练集可能不足以生成样本")
        else:
            print(f"  ✓ 训练集足够")

        print(f"\n✅ {name} 测试通过")

    print(f"\n{'='*60}")
    print("所有测试完成!")
    print(f"{'='*60}")

if __name__ == '__main__':
    test_data_files()
