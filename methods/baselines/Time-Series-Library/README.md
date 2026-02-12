# Time-Series-Library 最小集成

这是运行 FlowGNN_style 脚本所需的最小 Time-Series-Library 代码集。

## 目录结构

```
Time-Series-Library/
├── run.py              # 主运行脚本
├── data_provider/      # 数据加载器
├── models/             # 模型定义 (Autoformer, FEDformer, Mamba, 等)
├── layers/             # 网络层
├── exp/                # 实验框架
├── requirements.txt    # 依赖
└── scripts/            # 运行脚本 (FlowGNN_style)
    └── long_term_forecast/
        ├── metrla_script/    # METR-LA 数据集脚本
        └── pemsbay_script/   # PEMS-BAY 数据集脚本
```

## 依赖安装

```bash
cd methods/baselines/Time-Series-Library
pip install -r requirements.txt
```

主要依赖：
- torch >= 1.9.0
- numpy
- pandas
- matplotlib
- scikit-learn

## 数据位置

数据文件位于统一数据目录：
```
data/processed/Time-Series-Library/
├── metr-la.csv    # METR-LA 数据集 (207 节点, 70MB)
└── pems-bay.csv   # PEMS-BAY 数据集 (325 节点, 82MB)
```

数据格式：
- 第一列：date (时间戳)
- 中间列：各传感器节点的流量值
- 最后一列：OT (Output Target，Time-Series-Library 要求)

## 运行示例

### METR-LA 数据集

```bash
# Autoformer
cd scripts/long_term_forecast/metrla_script
./Autoformer_FlowGNN_style.sh

# TimesNet
./TimesNet_FlowGNN_style.sh

# Mamba
./Mamba_FlowGNN_style.sh
```

### PEMS-BAY 数据集

```bash
# Autoformer
cd scripts/long_term_forecast/pemsbay_script
./Autoformer_FlowGNN_style.sh

# FEDformer
./FEDformer_FlowGNN_style.sh
```

## 可用模型

METR-LA 和 PEMS-BAY 各有 8 个 FlowGNN_style 脚本：

1. **Autoformer** - Autoformer: Decomposition Transformers
2. **FEDformer** - Frequency Enhanced Decomposed Transformer
3. **Mamba** - Mamba: Linear-Time Sequence Modeling
4. **PatchTST** - Patch Time Series Transformer
5. **TimesNet** - TimesNet: Temporal 2D-Variation Modeling
6. **DLinear** - DLinear: Decomposition Linear
7. **Informer** - Informer: Beyond Efficient Transformer
8. **Transformer** - Vanilla Transformer

## FlowGNN_style 特点

这些脚本使用 FlowGNN 风格的数据预处理：
- `--scale_flow_only`: 仅标准化流量值
- `--no_overlap`: 输入和输出之间无重叠
- 从 144 步预测之后的 144 步 (144->144)

## 输出

训练结果会保存在：
- **检查点**: `checkpoints/long_term_forecast_[MODEL]_custom_[...]_[TIMESTAMP]/`
- **预测结果**: `results/long_term_forecast_[...]`
- **日志**: 自动记录训练损失和指标

## 注意事项

1. **GPU**: 建议使用 GPU 训练，通过 `export CUDA_VISIBLE_DEVICES=0` 设置
2. **内存**: PEMS-BAY 数据集较大 (325 节点)，建议减小 batch_size
3. **数据路径**: 脚本会自动查找数据，无需手动修改路径

## 原始仓库

完整代码和更多模型请参考：
https://github.com/thuml/Time-Series-Library

## 引用

```bibtex
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
