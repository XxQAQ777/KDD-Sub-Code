# Time-Series-Library 集成清单

## ✅ 已完成的工作

### 1. 代码集成

**位置**: `methods/baselines/Time-Series-Library/`

**复制的核心文件**:
- ✅ `run.py` - 主运行脚本 (14KB)
- ✅ `data_provider/` - 数据加载器目录
- ✅ `models/` - 所有模型定义 (41 个模型文件)
- ✅ `layers/` - 网络层实现
- ✅ `exp/` - 实验框架
- ✅ `requirements.txt` - 依赖清单
- ✅ `scripts/long_term_forecast/metrla_script/` - 8 个 METR-LA FlowGNN_style 脚本
- ✅ `scripts/long_term_forecast/pemsbay_script/` - 8 个 PEMS-BAY FlowGNN_style 脚本

**总大小**: 约 256MB (包含一些原始数据文件在 data_provider 中)

### 2. 数据准备

**位置**: `data/processed/Time-Series-Library/`

**数据文件**:
- ✅ `metr-la.csv` (70MB, 207 节点)
  - 格式: [date, sensor1, sensor2, ..., OT]
  - 时间范围: 2012/03/01 - 2012/06/30
  - 采样间隔: 5 分钟

- ✅ `pems-bay.csv` (82MB, 325 节点)
  - 格式: [date, sensor1, sensor2, ..., OT]
  - 时间范围: 2017/01/01 - 2017/06/30
  - 采样间隔: 5 分钟

**总大小**: 约 152MB

### 3. 脚本配置

所有 16 个 FlowGNN_style 脚本已配置：
- ✅ 自动查找 Time-Series-Library 代码 (run.py)
- ✅ 统一数据路径到 `data/processed/Time-Series-Library/`
- ✅ 移除所有绝对路径引用
- ✅ 可执行权限已设置

### 4. 文档

- ✅ `methods/baselines/Time-Series-Library/README.md` - 使用指南
- ✅ `INTEGRATION_LOG.md` - 已添加 Time-Series-Library 章节
- ✅ `README.md` - 已更新集成方法列表

## 📋 可用的模型

### METR-LA 数据集 (8 个模型)

1. **Autoformer_FlowGNN_style.sh**
2. **FEDformer_FlowGNN_style.sh**
3. **Mamba_FlowGNN_style.sh**
4. **PatchTST_FlowGNN_style.sh**
5. **TimesNet_FlowGNN_style.sh**
6. **Dlinear_FlowGNN_style.sh**
7. **Informer_FlowGNN_style.sh**
8. **Transformer_FlowGNN_style.sh**

### PEMS-BAY 数据集 (8 个模型)

相同的 8 个模型，针对 PEMS-BAY 数据集配置。

## 🚀 快速运行

### 安装依赖

```bash
cd methods/baselines/Time-Series-Library
pip install -r requirements.txt
```

主要依赖:
- torch >= 1.9.0
- numpy
- pandas
- matplotlib
- scikit-learn

### 运行示例

**METR-LA / Autoformer**:
```bash
cd methods/baselines/Time-Series-Library/scripts/long_term_forecast/metrla_script
./Autoformer_FlowGNN_style.sh
```

**PEMS-BAY / TimesNet**:
```bash
cd methods/baselines/Time-Series-Library/scripts/long_term_forecast/pemsbay_script
./TimesNet_FlowGNN_style.sh
```

## 📊 FlowGNN_style 参数说明

这些脚本使用特殊的 FlowGNN 风格预处理:

- `--scale_flow_only`: 仅标准化流量值
- `--no_overlap`: 输入和输出序列无重叠
- `--seq_len 144`: 输入序列长度 144 步
- `--pred_len 144`: 预测长度 144 步
- `--features M`: 多变量预测任务

## 🔍 代码查找机制

脚本会按以下顺序自动查找 Time-Series-Library 代码 (run.py):

1. `TrafficFM-main/extern/Time-Series-Library-main`
2. `TrafficFM-main/third_party/Time-Series-Library-main`
3. `TrafficFM-main/methods/baselines/Time-Series-Library`

当前已将代码放置在第 3 个位置。

## 💾 存储占用

- **代码**: ~256MB (methods/baselines/Time-Series-Library/)
- **数据**: ~152MB (data/processed/Time-Series-Library/)
- **总计**: ~408MB

## ⚠️ 注意事项

1. **GPU 推荐**: 建议使用 GPU 训练，通过 `export CUDA_VISIBLE_DEVICES=0` 设置
2. **内存管理**: PEMS-BAY 节点数较多 (325)，如内存不足可减小 batch_size
3. **数据格式**: CSV 文件最后一列必须命名为 "OT" (已自动处理)
4. **路径自动化**: 所有路径已配置为相对路径，无需手动修改

## 📁 完整目录结构

```
TrafficFM-main/
├── methods/baselines/Time-Series-Library/
│   ├── run.py
│   ├── README.md
│   ├── requirements.txt
│   ├── data_provider/
│   ├── models/
│   ├── layers/
│   ├── exp/
│   └── scripts/
│       └── long_term_forecast/
│           ├── metrla_script/
│           │   ├── Autoformer_FlowGNN_style.sh
│           │   ├── FEDformer_FlowGNN_style.sh
│           │   ├── Mamba_FlowGNN_style.sh
│           │   ├── PatchTST_FlowGNN_style.sh
│           │   ├── TimesNet_FlowGNN_style.sh
│           │   ├── Dlinear_FlowGNN_style.sh
│           │   ├── Informer_FlowGNN_style.sh
│           │   └── Transformer_FlowGNN_style.sh
│           └── pemsbay_script/
│               └── [相同的 8 个脚本]
└── data/processed/Time-Series-Library/
    ├── metr-la.csv
    └── pems-bay.csv
```

## ✨ 集成完成

所有必需的代码和数据已准备就绪，可以直接运行 FlowGNN_style 脚本！
