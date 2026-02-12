# 🎯 STGCN 测试脚本改进总结

## 📝 改进内容

### ✅ 1. 命令行参数支持

**原版本**（需要修改代码）:
```python
# 修改STGCN_Config.py
DATASET = 'PEMSBAY'
model_path = "../log/STGCN/PEMSBAY/20251229/model.pth"
```

**新版本**（命令行参数）:
```bash
python test_and_plot.py --dataset PEMSBAY --model_path /path/to/model.pth
```

### ✅ 2. 新增评估指标

- **原有指标**: MAE, MAPE, RMSE
- **新增指标**:
  - **CRPS** (Continuous Ranked Probability Score)
  - **Wasserstein Distance** (Earth Mover's Distance)

### ✅ 3. 增强的可视化

- 从 1x3 布局升级到 **2x3 布局**
- 新增 CRPS 和 WD 曲线图
- 新增所有指标的归一化对比图
- 散点图标题显示 MAE 和 WD

### ✅ 4. CSV 结果导出

**新增三个文件**:
- `metrics_detailed.csv`: 每个 horizon 的详细指标
- `metrics_summary.csv`: 统计摘要（均值、标准差、最小值、最大值）
- `metrics_summary.txt`: 可读的文本格式摘要

### ✅ 5. 批量测试脚本

**Shell 脚本** (`test_both_datasets.sh`):
```bash
bash test_both_datasets.sh cuda:0 64
```

**Python 脚本** (`batch_test.py`):
```bash
# 顺序测试
python batch_test.py --mode sequential

# 并行测试
python batch_test.py --mode parallel --devices cuda:0 cuda:1
```

### ✅ 6. 自动模型查找

如果不指定 `--model_path`，会自动查找最新的模型文件：
```bash
python test_and_plot.py --dataset METRLA  # 自动查找最新模型
```

### ✅ 7. 详细文档

- **TEST_USAGE.md**: 完整使用指南
- **QUICK_REFERENCE.md**: 快速参考卡片
- **README.md** (本文件): 改进总结

---

## 📊 新增评估指标说明

### CRPS (Continuous Ranked Probability Score)

**定义**: 衡量概率预测与实际观测之间的差异

**特点**:
- 对于确定性预测，CRPS = MAE
- 可以评估预测分布的质量
- 值越小越好

**意义**:
- 比MSE更全面地评估预测质量
- 适用于评估不确定性

### Wasserstein Distance (Earth Mover's Distance)

**定义**: 衡量两个概率分布之间的距离

**特点**:
- 也称为"地球移动距离"
- 考虑了分布的形状和位置
- 对异常值更加鲁棒

**意义**:
- 评估预测分布与真实分布的相似度
- 特别适用于多模态分布的评估
- 与 "Valley of Death" 现象相关

---

## 🎨 可视化改进对比

### 原版本（1x3布局）
```
[MAE] [MAPE] [RMSE]
```

### 新版本（2x3布局）
```
[MAE]  [MAPE]  [RMSE]
[CRPS] [WD]    [归一化对比]
```

**优势**:
- 更全面的指标展示
- 归一化对比图可以直观看出各指标趋势
- 更适合论文使用

---

## 📁 文件结构

```
FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN/
├── test_and_plot.py              # 主测试脚本（已改进）
├── batch_test.py                 # Python批量测试脚本（新增）
├── test_both_datasets.sh         # Shell批量测试脚本（新增）
├── TEST_USAGE.md                 # 完整使用文档（新增）
├── QUICK_REFERENCE.md            # 快速参考（新增）
└── README.md                     # 本文件（新增）
```

---

## 🚀 快速开始

### 单数据集测试

```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN

# METR-LA
python test_and_plot.py --dataset METRLA

# PEMS-BAY
python test_and_plot.py --dataset PEMSBAY
```

### 批量测试

```bash
# Shell脚本（简单）
bash test_both_datasets.sh

# Python脚本（灵活）
python batch_test.py --mode sequential
```

### 查看帮助

```bash
python test_and_plot.py --help
python batch_test.py --help
```

---

## 📈 使用示例

### 示例1: 快速测试两个数据集

```bash
# 顺序执行（约10-20分钟）
python batch_test.py --mode sequential
```

输出:
```
================================================================================
Sequential Testing Mode
================================================================================

================================================================================
Running: python test_and_plot.py --dataset METRLA --device cuda:0 --batch_size 64
================================================================================

Testing Results on Test Set
...
Horizon   1 | MAE: 2.84 | MAPE: 6.54 | RMSE: 5.31 | CRPS: 2.84 | WD: 0.23
...

✓ METRLA completed in 487.3s

...

✓ PEMSBAY completed in 523.1s

================================================================================
Testing Summary
================================================================================

METRLA      : ✓ PASS (487.3s)
PEMSBAY     : ✓ PASS (523.1s)
```

### 示例2: 指定模型路径测试

```bash
python test_and_plot.py \
    --dataset METRLA \
    --model_path ../log/STGCN/METRLA/20251228120000/METRLA_STGCN_best_model.pth \
    --device cuda:0
```

### 示例3: 并行测试（多GPU）

```bash
python batch_test.py \
    --mode parallel \
    --devices cuda:0 cuda:1 \
    --datasets METRLA PEMSBAY
```

---

## 📊 输出结果示例

### 目录结构
```
test_results_METRLA_20260115_143022/
├── metrics_over_horizons.png       # 6个子图的综合指标可视化
├── prediction_vs_groundtruth.png   # 4个scatter plots
├── time_series_predictions.png     # 3x5时间序列网格
├── error_distribution.png          # 4个误差分布直方图
├── spatial_error_heatmap.png       # 空间误差热力图
├── predictions.npy                 # Shape: (12, N_samples, 207, 1)
├── ground_truth.npy                # Shape: (12, N_samples, 207, 1)
├── metrics_detailed.csv            # 12行 x 6列
├── metrics_summary.csv             # 5行 x 5列
└── metrics_summary.txt             # 格式化的文本摘要
```

### metrics_detailed.csv 示例
```csv
Horizon,MAE,MAPE,RMSE,CRPS,Wasserstein_Distance
1,2.840000,6.540000,5.310000,2.840000,0.230000
2,3.120000,7.230000,5.890000,3.120000,0.289000
...
12,4.560000,11.230000,8.450000,4.560000,0.567000
```

### metrics_summary.csv 示例
```csv
Metric,Mean,Std,Min,Max
MAE,3.450000,0.580000,2.840000,4.560000
MAPE,8.230000,1.560000,6.540000,11.230000
RMSE,6.780000,1.120000,5.310000,8.450000
CRPS,3.450000,0.580000,2.840000,4.560000
Wasserstein_Distance,0.389000,0.112000,0.230000,0.567000
```

---

## 🔄 与原版本的兼容性

### 完全兼容

新版本**完全向后兼容**原版本的使用方式：

```bash
# 原版本使用方式（仍然有效）
cd STGCN
python test_and_plot.py  # 使用默认配置（PEMSBAY）

# 新版本额外支持
python test_and_plot.py --dataset METRLA  # 命令行切换数据集
```

### 数据格式兼容

- 输入数据格式：**完全相同**
- 模型加载方式：**完全相同**
- 输出.npy格式：**完全相同**
- 新增CSV输出：**不影响原有功能**

---

## 🎓 适用场景

### 科研实验

```bash
# 实验1: 对比两个数据集
python batch_test.py --mode sequential

# 实验2: 测试不同检查点
for checkpoint in ../log/STGCN/METRLA/*/best_model.pth; do
    python test_and_plot.py --dataset METRLA --model_path $checkpoint
done

# 实验3: 不同设备性能对比
python test_and_plot.py --dataset METRLA --device cuda:0
python test_and_plot.py --dataset METRLA --device cpu
```

### 论文制图

```bash
# 生成高质量图片（DPI=300）
python test_and_plot.py --dataset METRLA

# 导出数据用于额外分析
python
>>> import numpy as np
>>> pred = np.load('test_results_METRLA_xxx/predictions.npy')
>>> # 进行进一步的统计分析...
```

### 模型调试

```bash
# 快速测试新训练的模型
python test_and_plot.py \
    --dataset METRLA \
    --model_path ../log/STGCN/METRLA/latest/model.pth
```

---

## ⚙️ 技术细节

### 参数配置方式

```python
# setup_dataset_config() 函数中定义
configs = {
    'METRLA': {
        'dataset_dir': '../data/METR-LA/processed/',
        'graph_pkl': '../data/METR-LA/processed/adj_mx.pkl',
        'num_nodes': 207,
        'window': 12,
        'horizon': 12,
        # ...
    },
    'PEMSBAY': {
        # ...
    }
}
```

### 自动模型查找逻辑

```python
def find_latest_model(pattern):
    """查找最新的模型文件"""
    matching_files = glob.glob(pattern)
    matching_files.sort(key=os.path.getmtime, reverse=True)
    return matching_files[0]
```

### 指标计算

```python
# CRPS (对于确定性预测)
def compute_crps(pred, real):
    return torch.mean(torch.abs(pred - real)).item()

# Wasserstein Distance
def compute_wasserstein(pred, real):
    return wasserstein_distance(pred.flatten(), real.flatten())
```

---

## 🛠️ 故障排除

### 常见问题

**Q1: ImportError: No module named 'scipy'**
```bash
pip install scipy
```

**Q2: 找不到模型文件**
```bash
# 查找可用模型
find ../log/STGCN -name "*best_model.pth"

# 手动指定路径
python test_and_plot.py --dataset METRLA --model_path /path/to/model.pth
```

**Q3: CUDA out of memory**
```bash
# 减小batch size
python test_and_plot.py --dataset METRLA --batch_size 32

# 或使用CPU
python test_and_plot.py --dataset METRLA --device cpu
```

**Q4: 批量测试脚本权限问题**
```bash
chmod +x test_both_datasets.sh batch_test.py
```

---

## 📚 相关资源

### 文档
- **完整文档**: `TEST_USAGE.md`
- **快速参考**: `QUICK_REFERENCE.md`
- **Valley of Death 分析**: `../../motivation_valley_of_death/README.md`

### 脚本
- **主测试脚本**: `test_and_plot.py`
- **Shell批处理**: `test_both_datasets.sh`
- **Python批处理**: `batch_test.py`

### 数据
- **METR-LA**: `../data/METR-LA/processed/`
- **PEMS-BAY**: `../data/PEMS-BAY/processed/`

---

## 🎉 总结

### 主要改进

1. ✅ **命令行参数化** - 无需修改代码即可切换数据集
2. ✅ **新增评估指标** - CRPS & Wasserstein Distance
3. ✅ **增强可视化** - 从3个指标扩展到5个指标 + 综合对比
4. ✅ **CSV导出** - 便于后续分析和论文制表
5. ✅ **批量测试** - 支持顺序/并行测试多个数据集
6. ✅ **自动模型查找** - 可选的模型路径参数
7. ✅ **详细文档** - 完整的使用说明和快速参考

### 优势

- **灵活性**: 命令行参数控制，适应不同场景
- **全面性**: 5个评估指标，从多角度评估模型
- **易用性**: 批处理脚本，一键测试多个数据集
- **可视化**: 丰富的图表，直观展示模型性能
- **可追溯**: CSV导出，便于数据分析和对比

### 适用人群

- 🎓 **科研人员**: 需要全面评估模型性能
- 📊 **数据科学家**: 需要详细的性能分析
- 🏗️ **工程师**: 需要快速测试和部署
- 📝 **论文作者**: 需要高质量的可视化图表

---

## 📬 反馈与改进

如有问题或建议，欢迎反馈！

**常见改进方向**:
- 添加更多数据集支持
- 支持ensemble模型测试
- 添加confidence interval计算
- 支持自定义评估指标

---

**最后更新**: 2026-01-15
**版本**: 2.0
**作者**: Claude Code Assistant
