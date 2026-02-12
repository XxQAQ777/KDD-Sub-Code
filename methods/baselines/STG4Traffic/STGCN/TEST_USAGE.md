# STGCN Test and Plot - 使用说明

## 概述

`test_and_plot.py` 是一个灵活的测试和可视化脚本，支持通过命令行参数选择不同的数据集进行测试。

## 新增功能

- ✅ **命令行参数支持**：可以通过命令行选择数据集
- ✅ **自动配置**：根据数据集自动配置相关路径和参数
- ✅ **模型路径灵活指定**：支持手动指定或自动查找最新模型
- ✅ **多数据集支持**：METR-LA 和 PEMS-BAY

## 命令行参数

### 必需参数

无（所有参数都有默认值）

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | PEMSBAY | 数据集选择：METRLA 或 PEMSBAY |
| `--model_path` | str | None | 模型检查点路径（如不指定，自动查找最新模型）|
| `--device` | str | cuda:0 | 计算设备：cuda:0, cuda:1, cpu 等 |
| `--batch_size` | int | 64 | 测试批大小 |

## 使用示例

### 1. 测试 PEMS-BAY 数据集（默认）

```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN

# 使用默认设置（会自动查找最新的PEMS-BAY模型）
python test_and_plot.py
```

### 2. 测试 METR-LA 数据集

```bash
# 使用默认设置（会自动查找最新的METR-LA模型）
python test_and_plot.py --dataset METRLA
```

### 3. 指定模型路径

```bash
# 为PEMS-BAY指定具体的模型文件
python test_and_plot.py \
    --dataset PEMSBAY \
    --model_path ../log/STGCN/PEMSBAY/20251229012414/PEMSBAY_STGCN_best_model.pth

# 为METR-LA指定具体的模型文件
python test_and_plot.py \
    --dataset METRLA \
    --model_path ../log/STGCN/METRLA/20251228120000/METRLA_STGCN_best_model.pth
```

### 4. 指定GPU设备

```bash
# 使用GPU 1
python test_and_plot.py --dataset PEMSBAY --device cuda:1

# 使用CPU
python test_and_plot.py --dataset METRLA --device cpu
```

### 5. 完整命令示例

```bash
# 测试METR-LA数据集，使用GPU 0，指定模型路径，batch size为32
python test_and_plot.py \
    --dataset METRLA \
    --device cuda:0 \
    --batch_size 32 \
    --model_path ../log/STGCN/METRLA/20251228/METRLA_STGCN_best_model.pth
```

## 数据集自动配置

脚本会根据 `--dataset` 参数自动配置以下内容：

### METR-LA 配置
```python
{
    'dataset_dir': '../data/METR-LA/processed/',
    'graph_pkl': '../data/METR-LA/processed/adj_mx.pkl',
    'num_nodes': 207,
    'window': 12,
    'horizon': 12,
    'input_dim': 1,
    'KS': 3,
    'KT': 3,
    'channels': [64, 32, 64],
    'dropout': 0.3
}
```

### PEMS-BAY 配置
```python
{
    'dataset_dir': '../data/PEMS-BAY/processed/',
    'graph_pkl': '../data/PEMS-BAY/processed/adj_mx_bay.pkl',
    'num_nodes': 325,
    'window': 12,
    'horizon': 12,
    'input_dim': 1,
    'KS': 3,
    'KT': 3,
    'channels': [64, 32, 64],
    'dropout': 0.3
}
```

## 输出结果

脚本会在当前目录创建一个时间戳命名的文件夹，包含：

```
test_results_{DATASET}_{TIMESTAMP}/
├── metrics_over_horizons.png          # 5个指标曲线图（2x3布局）
├── prediction_vs_groundtruth.png      # 预测vs真实值散点图
├── time_series_predictions.png         # 时间序列预测可视化
├── error_distribution.png              # 误差分布直方图
├── spatial_error_heatmap.png           # 空间误差热力图
├── predictions.npy                     # 预测结果（NumPy数组）
├── ground_truth.npy                    # 真实值（NumPy数组）
├── metrics_detailed.csv                # 详细指标（每个horizon）
├── metrics_summary.csv                 # 统计摘要
└── metrics_summary.txt                 # 文本格式摘要
```

## 评估指标

脚本计算以下5个指标：

1. **MAE** (Mean Absolute Error): 平均绝对误差
2. **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差
3. **RMSE** (Root Mean Square Error): 均方根误差
4. **CRPS** (Continuous Ranked Probability Score): 连续排序概率分数
5. **WD** (Wasserstein Distance): Wasserstein距离（地球移动距离）

## 错误处理

### 模型文件未找到

如果模型文件不存在，脚本会提示：

```
Error: Model file not found!
  Searched pattern: ../log/STGCN/PEMSBAY/*/PEMSBAY_STGCN_best_model.pth

Please specify the model path using --model_path argument.
Example:
  python test_and_plot.py --dataset PEMSBAY --model_path /path/to/model.pth
```

**解决方法**：
1. 使用 `--model_path` 指定正确的模型路径
2. 或者确保训练好的模型在默认路径下

### 数据集路径不存在

如果数据集目录不存在，脚本会报错。请确保：
- METR-LA 数据在 `../data/METR-LA/processed/`
- PEMS-BAY 数据在 `../data/PEMS-BAY/processed/`

## 快速对比两个数据集

```bash
# 终端1：测试METR-LA
python test_and_plot.py --dataset METRLA --device cuda:0

# 终端2：测试PEMS-BAY（如果有多GPU）
python test_and_plot.py --dataset PEMSBAY --device cuda:1
```

或者顺序执行：

```bash
# 先测试METR-LA
python test_and_plot.py --dataset METRLA
echo "METR-LA testing completed!"

# 再测试PEMS-BAY
python test_and_plot.py --dataset PEMSBAY
echo "PEMS-BAY testing completed!"
```

## 常见问题

### Q1: 如何查看所有可用参数？

```bash
python test_and_plot.py --help
```

### Q2: 可以添加新的数据集吗？

可以！在 `setup_dataset_config()` 函数中添加新的数据集配置：

```python
configs = {
    'METRLA': {...},
    'PEMSBAY': {...},
    'NEWDATASET': {  # 添加你的新数据集
        'dataset_dir': '../data/NEWDATASET/processed/',
        'graph_pkl': '../data/NEWDATASET/processed/adj_mx.pkl',
        'num_nodes': 123,
        # ... 其他配置
    }
}
```

然后在 `parse_args()` 的 `choices` 中添加新数据集名称。

### Q3: 如何修改默认参数？

直接在命令行中覆盖即可：

```bash
python test_and_plot.py --dataset METRLA --batch_size 128 --device cuda:1
```

## 注意事项

1. **模型兼容性**：确保模型是用相应数据集训练的
2. **内存使用**：大batch_size可能导致GPU内存不足
3. **路径正确性**：相对路径基于脚本所在目录
4. **设备可用性**：使用CUDA前确保GPU可用

## 与原版本的区别

### 原版本（硬编码）
```python
# 需要手动修改代码
DATASET = 'PEMSBAY'
model_path = "../log/STGCN/PEMSBAY/20251229012414/PEMSBAY_STGCN_best_model.pth"
```

### 新版本（命令行参数）
```bash
# 灵活的命令行控制
python test_and_plot.py --dataset PEMSBAY --model_path path/to/model.pth
```

## 总结

新版本的 `test_and_plot.py` 提供了更灵活的使用方式：

- ✅ 无需修改代码即可切换数据集
- ✅ 支持自动查找最新模型
- ✅ 命令行参数化，易于批处理
- ✅ 更好的错误提示和用户体验
- ✅ 保持与原有功能的完全兼容

快速开始：

```bash
# METR-LA
python test_and_plot.py --dataset METRLA

# PEMS-BAY
python test_and_plot.py --dataset PEMSBAY
```
