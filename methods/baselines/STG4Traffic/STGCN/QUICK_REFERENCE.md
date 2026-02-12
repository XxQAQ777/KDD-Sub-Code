# STGCN 测试快速参考

## 🚀 基本用法

### 单个数据集测试

```bash
# METR-LA
python test_and_plot.py --dataset METRLA

# PEMS-BAY
python test_and_plot.py --dataset PEMSBAY
```

### 指定模型路径

```bash
python test_and_plot.py \
    --dataset METRLA \
    --model_path ../log/STGCN/METRLA/20251228/METRLA_STGCN_best_model.pth
```

### 指定GPU

```bash
# 使用 GPU 0
python test_and_plot.py --dataset METRLA --device cuda:0

# 使用 GPU 1
python test_and_plot.py --dataset PEMSBAY --device cuda:1

# 使用 CPU
python test_and_plot.py --dataset METRLA --device cpu
```

---

## 🔄 批量测试

### 方法1: Shell脚本（顺序执行）

```bash
# 使用默认设置（cuda:0, batch_size=64）
bash test_both_datasets.sh

# 指定GPU
bash test_both_datasets.sh cuda:1

# 指定GPU和batch size
bash test_both_datasets.sh cuda:0 128
```

### 方法2: Python脚本（更灵活）

```bash
# 顺序测试两个数据集
python batch_test.py --mode sequential

# 并行测试（需要多GPU）
python batch_test.py --mode parallel --devices cuda:0 cuda:1

# 只测试METR-LA
python batch_test.py --datasets METRLA

# 只测试PEMS-BAY
python batch_test.py --datasets PEMSBAY

# 自定义batch size
python batch_test.py --mode sequential --batch_size 128
```

---

## 📊 完整参数列表

### test_and_plot.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | PEMSBAY | 数据集: METRLA 或 PEMSBAY |
| `--model_path` | None | 模型路径（可选，自动查找最新）|
| `--device` | cuda:0 | 计算设备 |
| `--batch_size` | 64 | 测试批大小 |

### batch_test.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--datasets` | METRLA PEMSBAY | 要测试的数据集列表 |
| `--mode` | sequential | 测试模式: sequential 或 parallel |
| `--device` | cuda:0 | 顺序模式使用的设备 |
| `--devices` | cuda:0 cuda:1 | 并行模式使用的设备列表 |
| `--batch_size` | 64 | 批大小 |
| `--model_paths` | [] | 模型路径列表（可选）|

---

## 🎯 常见场景

### 场景1: 快速测试一个数据集

```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN
python test_and_plot.py --dataset METRLA
```

### 场景2: 测试两个数据集并对比

```bash
# 方法1: 顺序执行
python batch_test.py --mode sequential

# 方法2: 并行执行（推荐，如果有多GPU）
python batch_test.py --mode parallel --devices cuda:0 cuda:1
```

### 场景3: 指定特定模型进行测试

```bash
python test_and_plot.py \
    --dataset PEMSBAY \
    --model_path ../log/STGCN/PEMSBAY/20251229012414/PEMSBAY_STGCN_best_model.pth
```

### 场景4: 内存受限，减小batch size

```bash
python test_and_plot.py --dataset METRLA --batch_size 32
```

### 场景5: 在多台机器上分布式测试

```bash
# 机器1: 测试METR-LA
ssh server1 "cd /path/to/STGCN && python test_and_plot.py --dataset METRLA"

# 机器2: 测试PEMS-BAY
ssh server2 "cd /path/to/STGCN && python test_and_plot.py --dataset PEMSBAY"
```

---

## 📁 输出结果

每次测试会创建一个时间戳命名的目录：

```
test_results_{DATASET}_{TIMESTAMP}/
├── metrics_over_horizons.png       # 指标曲线（5个）
├── prediction_vs_groundtruth.png   # 散点图
├── time_series_predictions.png     # 时间序列
├── error_distribution.png          # 误差分布
├── spatial_error_heatmap.png       # 空间热力图
├── predictions.npy                 # 预测结果
├── ground_truth.npy                # 真实值
├── metrics_detailed.csv            # 详细指标
├── metrics_summary.csv             # 统计摘要
└── metrics_summary.txt             # 文本摘要
```

---

## 📈 评估指标

| 指标 | 说明 |
|------|------|
| MAE | 平均绝对误差 |
| MAPE | 平均绝对百分比误差 |
| RMSE | 均方根误差 |
| CRPS | 连续排序概率分数 |
| WD | Wasserstein距离 |

---

## ⚠️ 故障排除

### 问题: 模型文件未找到

```bash
Error: Model file not found!
```

**解决方法**:
```bash
# 使用 --model_path 指定路径
python test_and_plot.py --dataset METRLA --model_path /path/to/model.pth

# 或者查找可用的模型
find ../log/STGCN -name "*_best_model.pth"
```

### 问题: GPU内存不足

```bash
RuntimeError: CUDA out of memory
```

**解决方法**:
```bash
# 减小batch size
python test_and_plot.py --dataset METRLA --batch_size 32

# 或使用CPU
python test_and_plot.py --dataset METRLA --device cpu
```

### 问题: 数据集路径不存在

```bash
FileNotFoundError: [Errno 2] No such file or directory
```

**解决方法**:
```bash
# 检查数据是否存在
ls ../data/METR-LA/processed/
ls ../data/PEMS-BAY/processed/

# 如果不存在，运行数据预处理
cd ../data
python preprocess.py
```

---

## 🔧 高级用法

### 自动化测试脚本

```bash
#!/bin/bash
# auto_test.sh - 自动化测试多个检查点

DATASETS=("METRLA" "PEMSBAY")
CHECKPOINTS=(
    "../log/STGCN/METRLA/run1/METRLA_STGCN_best_model.pth"
    "../log/STGCN/PEMSBAY/run1/PEMSBAY_STGCN_best_model.pth"
)

for i in ${!DATASETS[@]}; do
    echo "Testing ${DATASETS[$i]}..."
    python test_and_plot.py \
        --dataset ${DATASETS[$i]} \
        --model_path ${CHECKPOINTS[$i]} \
        --device cuda:0
done
```

### 并行测试（使用GNU parallel）

```bash
# 如果安装了GNU parallel
parallel python test_and_plot.py --dataset {} ::: METRLA PEMSBAY
```

---

## 📚 相关文档

- **完整文档**: `TEST_USAGE.md`
- **代码**: `test_and_plot.py`
- **批处理脚本**: `batch_test.py`, `test_both_datasets.sh`

---

## 💡 提示

1. **首次运行**: 使用默认设置测试，确保环境正确
2. **批量测试**: 使用 `batch_test.py` 更方便
3. **结果对比**: 查看生成的CSV文件对比不同数据集
4. **可视化**: 查看PNG图片直观了解模型性能
5. **并行加速**: 如有多GPU，使用 `--mode parallel`

---

## ⭐ 快速开始

```bash
# 1. 进入目录
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN

# 2. 测试单个数据集
python test_and_plot.py --dataset METRLA

# 3. 批量测试
python batch_test.py --mode sequential

# 4. 查看结果
ls -lh test_results_*/
```

搞定！🎉
