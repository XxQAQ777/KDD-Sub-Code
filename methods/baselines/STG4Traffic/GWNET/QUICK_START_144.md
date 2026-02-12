# GWNET 144时间步训练快速指南

## 一、修改内容总结

### 1. 已修改的文件
- `GWNET_Config_144.py`: 数据路径已修正
  - `dataset_dir` → `../data/METR-LA/processed/`
  - `graph_pkl` → `../data/METR-LA/processed/adj_mx.pkl`
  - `DEVICE` → `cuda:0` (可根据需要修改)

### 2. 新增的文件
- `prepare_data_144.py`: 数据准备脚本
- `train_gwnet_144.py`: 训练脚本（带详细日志）
- `run_gwnet_144.sh`: 一键运行脚本
- `README_144.md`: 详细说明文档
- `QUICK_START_144.md`: 本文档

## 二、执行步骤

### 选项A: 一键运行（推荐）
```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/GWNET
./run_gwnet_144.sh
```

### 选项B: 分步运行
```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/GWNET

# 步骤1: 生成144时间步数据
python prepare_data_144.py

# 步骤2: 训练模型
python train_gwnet_144.py
```

### 选项C: 使用原始脚本
```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/GWNET

# 步骤1: 生成数据（使用根目录的脚本）
cd ..
python generate_metr_la_data_144.py --dataset METR-LA

# 步骤2: 训练模型
cd GWNET
python GWNET_Main_144.py
```

## 三、重要配置

### GPU设置
编辑 `GWNET_Config_144.py`:
```python
DEVICE = 'cuda:0'  # 改为你的GPU编号
```

### Batch Size调整
如果遇到显存不足，编辑 `METRLA_GWNET_144.conf`:
```ini
[train]
batch_size = 8  # 从16减小到8或更小
```

## 四、预期输出

### 数据准备输出
```
生成数据...
x shape: (23974, 144, 207, 2)
y shape: (23974, 144, 207, 2)
train x: (16781, 144, 207, 2)
val x: (2397, 144, 207, 2)
test x: (4796, 144, 207, 2)
```

### 训练输出
```
Epoch 001, Train Loss: 3.2145, Valid Loss: 4.1523, Valid RMSE: 5.1234, Valid MAPE: 0.1234
Epoch 002, Train Loss: 2.8432, Valid Loss: 3.9234, Valid RMSE: 4.9123, Valid MAPE: 0.1156
...
```

## 五、验证数据

检查数据是否正确生成:
```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed
python -c "
import numpy as np
data = np.load('data/METR-LA/processed/train.npz')
print('x_train shape:', data['x'].shape)
print('y_train shape:', data['y'].shape)
print('Expected: (samples, 144, 207, 2)')
"
```

## 六、常见问题

**Q: 运行时显示 "No module named 'xxx'"**
A: 安装缺失的依赖包
```bash
pip install numpy pandas torch h5py
```

**Q: CUDA out of memory**
A: 减小batch_size或使用CPU
```python
# GWNET_Config_144.py
DEVICE = 'cpu'  # 使用CPU
# 或者
# METRLA_GWNET_144.conf
batch_size = 4  # 减小batch size
```

**Q: 找不到数据文件**
A: 确保先运行数据准备步骤

**Q: 训练太慢**
A: 检查是否使用GPU
```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 七、完整代码示例

### 训练脚本
```python
# 在GWNET目录下运行
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/GWNET
python train_gwnet_144.py
```

### 测试脚本
```python
# 1. 修改 GWNET_Config_144.py
MODE = 'test'

# 2. 在 train_gwnet_144.py 或 GWNET_Main_144.py 中指定checkpoint
checkpoint = "../log/GWNET/METRLA_144/20240104120000/METRLA_GWNET_best_model.pth"

# 3. 运行测试
python train_gwnet_144.py
```

## 八、文件说明

| 文件 | 作用 |
|------|------|
| `GWNET_Config_144.py` | 配置参数 |
| `METRLA_GWNET_144.conf` | 模型和训练超参数 |
| `prepare_data_144.py` | 生成144时间步数据 |
| `train_gwnet_144.py` | 训练主程序（新版，带详细输出） |
| `GWNET_Main_144.py` | 训练主程序（原版） |
| `run_gwnet_144.sh` | 一键运行脚本 |

## 九、关键参数对比

| 参数 | 12时间步 | 144时间步 | 说明 |
|------|---------|-----------|------|
| window | 12 | 144 | 输入窗口 |
| horizon | 12 | 144 | 预测窗口 |
| batch_size | 64 | 16 | 批次大小（内存限制） |
| blocks | 4 | 6 | 模型深度 |
| layers | 2 | 3 | 每块层数 |
| lr_decay | False | True | 学习率衰减 |

## 十、模型输入输出

```
输入 (Input):
  - 形状: (batch_size, 144, 207, 2)
  - 含义: [批次, 时间步, 节点数, 特征(速度+时间)]

输出 (Output):
  - 形状: (batch_size, 144, 207, 1)
  - 含义: [批次, 预测时间步, 节点数, 速度预测值]
```

## 联系方式

如有问题，请检查日志文件：
```
../log/GWNET/METRLA_144/*/run.log
```
