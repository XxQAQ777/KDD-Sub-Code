# STG4Traffic 集成说明

本文档说明 STG4Traffic 中的 DCRNN, DGCRN, GWNET, STGCN 四个模型在 TrafficFM-main 中的集成情况。

## 目录结构

```
TrafficFM-main/methods/baselines/STG4Traffic/
├── path_config.py          # 统一路径配置
├── lib/                     # 共享库
│   ├── data_loader.py
│   ├── utils.py
│   └── generate_adj_mx.py
├── model/                   # 模型定义
│   ├── DCRNN/
│   ├── DGCRN/
│   ├── GWNET/
│   └── STGCN/
├── DCRNN/                   # DCRNN 训练代码
│   ├── DCRNN_Config.py     # 已修改
│   ├── DCRNN_Main.py
│   └── *.conf
├── DGCRN/                   # DGCRN 训练代码
│   ├── DGCRN_Config.py     # 已修改
│   ├── DGCRN_Main.py
│   └── *.conf
├── GWNET/                   # GWNET 训练代码
│   ├── GWNET_Config.py     # 已修改
│   ├── GWNET_Main.py
│   └── *.conf
└── STGCN/                   # STGCN 训练代码
    ├── STGCN_Config.py     # 已修改
    ├── STGCN_Main.py
    └── *.conf
```

## 数据路径

数据统一存放在：
```
TrafficFM-main/data/processed/STG4Traffic/
├── METR-LA/processed/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── adj_mx.pkl
└── PEMS-BAY/processed/
    ├── train.npz
    ├── val.npz
    ├── test.npz
    └── adj_mx_bay.pkl
```

## 路径配置说明

所有四个模型通过 `path_config.py` 统一管理数据路径：

```python
# path_config.py 自动计算路径
STG4TRAFFIC_ROOT = TrafficFM-main/methods/baselines/STG4Traffic
TRAFFICFM_ROOT = TrafficFM-main
DATA_ROOT = TrafficFM-main/data/processed/STG4Traffic

METRLA_DIR = DATA_ROOT/METR-LA/processed
PEMSBAY_DIR = DATA_ROOT/PEMS-BAY/processed
METRLA_GRAPH = METRLA_DIR/adj_mx.pkl
PEMSBAY_GRAPH = PEMSBAY_DIR/adj_mx_bay.pkl
```

## 快速运行

### 1. DCRNN

```bash
cd TrafficFM-main/methods/baselines/STG4Traffic/DCRNN

# 编辑 DCRNN_Config.py，修改 DATASET 变量
# DATASET = 'METRLA'  # 或 'PEMSBAY'

python DCRNN_Main.py
```

### 2. DGCRN

```bash
cd TrafficFM-main/methods/baselines/STG4Traffic/DGCRN

# 编辑 DGCRN_Config.py，修改 DATASET 变量
# DATASET = 'METRLA'  # 或 'PEMSBAY'

python DGCRN_Main.py
```

### 3. GWNET

```bash
cd TrafficFM-main/methods/baselines/STG4Traffic/GWNET

# 编辑 GWNET_Config.py，修改 DATASET 变量
# DATASET = 'METRLA'  # 或 'PEMSBAY'

python GWNET_Main.py
```

### 4. STGCN

```bash
cd TrafficFM-main/methods/baselines/STG4Traffic/STGCN

# 编辑 STGCN_Config.py，修改 DATASET 变量
# DATASET = 'METRLA'  # 或 'PEMSBAY'

python STGCN_Main.py
```

## 关键修改

### 修改内容

四个模型的 Config.py 文件都进行了相同的修改，以 DCRNN 为例：

**原始代码**:
```python
DATASET = 'PEMSBAY'
dataset_dir = "../data/PEMS-BAY/processed/"
graph_pkl = "../data/PEMS-BAY/processed/adj_mx_bay.pkl"
```

**修改后**:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import METRLA_DIR, PEMSBAY_DIR, METRLA_GRAPH, PEMSBAY_GRAPH

DATASET = 'PEMSBAY'

if DATASET == 'METRLA':
    dataset_dir = METRLA_DIR
    graph_pkl = METRLA_GRAPH
else:
    dataset_dir = PEMSBAY_DIR
    graph_pkl = PEMSBAY_GRAPH
```

### 验证路径

运行路径配置验证：

```bash
cd TrafficFM-main/methods/baselines/STG4Traffic
python path_config.py
```

预期输出:
```
STG4Traffic Path Configuration:
  STG4TRAFFIC_ROOT: TrafficFM-main/methods/baselines/STG4Traffic
  TRAFFICFM_ROOT: TrafficFM-main
  DATA_ROOT: TrafficFM-main/data/processed/STG4Traffic
  METRLA_DIR: .../data/processed/STG4Traffic/METR-LA/processed
  PEMSBAY_DIR: .../data/processed/STG4Traffic/PEMS-BAY/processed
  ...
```

## 输出位置

训练日志和模型保存在各自的目录下：

```
methods/baselines/STG4Traffic/log/
├── DCRNN/
│   ├── METRLA/
│   └── PEMSBAY/
├── DGCRN/
│   ├── METRLA/
│   └── PEMSBAY/
├── GWNET/
│   ├── METRLA/
│   └── PEMSBAY/
└── STGCN/
    ├── METRLA/
    └── PEMSBAY/
```

## 注意事项

1. **数据集切换**: 修改对应 Config.py 中的 `DATASET` 变量
2. **GPU 设置**: 修改 Config.py 中的 `DEVICE` 变量
3. **训练/测试模式**: 修改 Config.py 中的 `MODE` 变量
4. **共享库**: 四个模型共享 `lib/` 和 `model/` 目录
5. **独立运行**: 每个模型可以独立运行，互不干扰

## 常见问题

### Q: 如何切换数据集？

A: 编辑对应模型的 Config.py 文件，修改 `DATASET` 变量：
```python
DATASET = 'METRLA'  # 或 'PEMSBAY'
```

### Q: 如何修改 GPU？

A: 编辑 Config.py 文件，修改 `DEVICE` 变量：
```python
DEVICE = 'cuda:0'  # 或 'cuda:1', 'cuda:2', etc.
```

### Q: 数据文件找不到？

A: 运行 `python path_config.py` 检查路径配置是否正确，确保数据文件存在于：
- `data/processed/STG4Traffic/METR-LA/processed/`
- `data/processed/STG4Traffic/PEMS-BAY/processed/`

### Q: 如何查看训练日志？

A: 日志保存在 `methods/baselines/STG4Traffic/log/[MODEL]/[DATASET]/` 目录下。

## 更多信息

详细配置和参数说明见 [INTEGRATION_LOG.md](../../INTEGRATION_LOG.md#stg4traffic-dcrnn-dgcrn-gwnet-stgcn)。
