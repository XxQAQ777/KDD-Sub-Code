#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证数据准备是否正确
"""

import numpy as np
import os

def verify_dataset(dataset_name, dataset_path):
    """验证单个数据集"""
    print(f"\n{'='*60}")
    print(f"验证数据集: {dataset_name}")
    print(f"{'='*60}")

    # 检查文件是否存在
    flow_path = os.path.join(dataset_path, 'flow.npy')
    adj_path = os.path.join(dataset_path, 'adj.npy')

    if not os.path.exists(flow_path):
        print(f"❌ 错误: flow.npy 不存在于 {dataset_path}")
        return False

    if not os.path.exists(adj_path):
        print(f"❌ 错误: adj.npy 不存在于 {dataset_path}")
        return False

    # 加载数据
    flow = np.load(flow_path)
    adj = np.load(adj_path)

    print(f"✓ Flow shape: {flow.shape}")
    print(f"✓ Adj shape: {adj.shape}")

    # 验证维度
    T, V, D = flow.shape
    assert adj.shape == (V, V), f"邻接矩阵维度不匹配: {adj.shape} vs ({V}, {V})"
    assert D == 3, f"特征维度应该是3，实际是{D}"

    print(f"\n数据统计:")
    print(f"  时间步数 (T): {T}")
    print(f"  节点数 (V): {V}")
    print(f"  特征数 (D): {D}")

    # 检查每个特征的范围
    print(f"\n特征统计:")
    print(f"  Feature 0 (标准化流量):")
    print(f"    - 范围: [{flow[:,:,0].min():.4f}, {flow[:,:,0].max():.4f}]")
    print(f"    - 均值: {flow[:,:,0].mean():.4f}")
    print(f"    - 标准差: {flow[:,:,0].std():.4f}")

    print(f"  Feature 1 (tod):")
    print(f"    - 范围: [{flow[:,:,1].min():.0f}, {flow[:,:,1].max():.0f}]")
    print(f"    - 唯一值数量: {len(np.unique(flow[:,:,1]))}")

    print(f"  Feature 2 (dow):")
    print(f"    - 范围: [{flow[:,:,2].min():.0f}, {flow[:,:,2].max():.0f}]")
    print(f"    - 唯一值数量: {len(np.unique(flow[:,:,2]))}")

    # 检查邻接矩阵
    print(f"\n邻接矩阵统计:")
    print(f"  - 范围: [{adj.min():.4f}, {adj.max():.4f}]")
    print(f"  - 非零元素比例: {(adj != 0).sum() / adj.size * 100:.2f}%")
    print(f"  - 对角线元素: {np.diag(adj)[:5]} ...")

    # 验证数据划分
    train_size = int(T * 0.7)
    val_size = int(T * 0.1)
    test_size = T - train_size - val_size

    print(f"\n数据划分 (7:1:2):")
    print(f"  - 训练集: 0 到 {train_size} ({train_size/T*100:.1f}%)")
    print(f"  - 验证集: {train_size} 到 {train_size+val_size} ({val_size/T*100:.1f}%)")
    print(f"  - 测试集: {train_size+val_size} 到 {T} ({test_size/T*100:.1f}%)")

    # 检查训练集的标准化
    train_flow = flow[:train_size, :, 0]
    print(f"\n训练集标准化验证:")
    print(f"  - 训练集流量均值: {train_flow.mean():.4f} (应接近0)")
    print(f"  - 训练集流量标准差: {train_flow.std():.4f} (应接近1)")

    # 检查是否有NaN或Inf
    if np.isnan(flow).any():
        print(f"⚠️  警告: flow数据中包含NaN值")
    if np.isinf(flow).any():
        print(f"⚠️  警告: flow数据中包含Inf值")
    if np.isnan(adj).any():
        print(f"⚠️  警告: adj数据中包含NaN值")
    if np.isinf(adj).any():
        print(f"⚠️  警告: adj数据中包含Inf值")

    print(f"\n✅ {dataset_name} 数据集验证通过!")
    return True

def main():
    base_dir = '/home/xiaoxiao/DiffSTG-main/data/dataset'

    # 验证METR-LA
    metr_la_path = os.path.join(base_dir, 'metr-la')
    verify_dataset('METR-LA', metr_la_path)

    # 验证PEMS-BAY
    pems_bay_path = os.path.join(base_dir, 'pems-bay')
    verify_dataset('PEMS-BAY', pems_bay_path)

    print(f"\n{'='*60}")
    print("所有数据集验证完成!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
