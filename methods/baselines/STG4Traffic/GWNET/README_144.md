# GWNET 144->144时间步预测配置指南

## 概述

本文档说明如何配置和训练GWNET模型进行144个时间步到144个时间步的预测任务。

## 主要修改点

### 1. 数据配置
- **输入窗口 (window)**: 144个时间步
- **预测窗口 (horizon)**: 144个时间步
- **数据维度**: `(batch_size, 144, num_nodes, features)`

### 2. 模型配置
为了处理更长的时间序列，增加了模型容量：
- **Blocks**: 6 (原来是4)
- **Layers**: 3 (原来是2)
- **Batch size**: 16 (原来是64，因为序列更长需要更多内存)

### 3. 训练配置
- **学习率衰减**: 启用，步骤为 [20, 40, 60, 80]
- **早停耐心**: 20个epoch
- **课程学习**: 禁用 (cl=False)

## 快速开始

### 方法1: 使用一键脚本（推荐）

```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/GWNET
chmod +x run_gwnet_144.sh
./run_gwnet_144.sh
```

### 方法2: 分步执行

#### 步骤1: 准备数据

```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/GWNET
python prepare_data_144.py
```

#### 步骤2: 训练模型

```bash
python train_gwnet_144.py
```

## 配置参数说明

### 关键参数 (METRLA_GWNET_144.conf)

- `window = 144` - 输入时间窗口
- `horizon = 144` - 预测时间窗口
- `blocks = 6` - 模型块数
- `layers = 3` - 每块层数
- `batch_size = 16` - 批次大小

## 常见问题

### Q1: CUDA out of memory
减小batch_size到8或更小

### Q2: 数据文件不存在
先运行 `prepare_data_144.py` 生成数据

### Q3: 如何更改GPU
修改 `GWNET_Config_144.py` 中的 `DEVICE = 'cuda:X'`

## 性能监控

训练日志保存在:
```
../log/GWNET/METRLA_144/YYYYMMDDHHMMSS/
```
  cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/GWNET
  python GWNET_Main.py --dataset PEMSBAY --dataset_dir ../data/PEMS-BAY/processed/ --graph_pkl ../data/PEMS-BAY/processed/adj_mx.pkl