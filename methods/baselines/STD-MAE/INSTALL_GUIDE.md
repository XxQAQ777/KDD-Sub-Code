# STD-MAE 虚拟环境依赖安装指南

## 方法 1: 使用安装脚本（推荐）

我已经创建了一个自动安装脚本，您可以直接运行：

```bash
cd /home/xiaoxiao/STD-MAE-main
bash install_dependencies.sh
```

## 方法 2: 手动安装

### 步骤 1: 激活虚拟环境

```bash
cd /home/xiaoxiao/STD-MAE-main
source venv_stdmae/bin/activate
```

### 步骤 2: 安装 requirements.txt 中的依赖

```bash
pip install -r requirements.txt
```

这将安装以下包：
- easy_torch==1.2.12
- easydict==1.10
- numpy==1.21.5
- positional_encodings==6.0.1
- scikit_learn==1.0.2
- scipy==1.7.3
- setproctitle==1.3.2
- sympy==1.10.1
- timm==0.6.11
- torch==1.13.1
- torch_summary==1.4.5
- tvm==1.0.0

### 步骤 3: 安装额外依赖

```bash
pip install pandas tables
```

这两个包是数据处理所需的：
- **pandas**: 用于读取 HDF5 数据文件
- **tables**: 用于处理 .h5 文件格式

### 步骤 4: 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import tables; print(f'Tables: {tables.__version__}')"
python -c "import easydict; print('EasyDict: OK')"
python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
```

如果所有命令都成功执行，说明依赖安装完成。

## 方法 3: 分步安装（如果遇到问题）

如果一次性安装遇到问题，可以分步安装：

```bash
# 激活虚拟环境
source venv_stdmae/bin/activate

# 核心依赖
pip install numpy==1.21.5
pip install scipy==1.7.3
pip install torch==1.13.1

# 工具库
pip install easydict==1.10
pip install setproctitle==1.3.2
pip install sympy==1.10.1

# 机器学习相关
pip install scikit_learn==1.0.2
pip install timm==0.6.11
pip install torch_summary==1.4.5

# 特殊依赖
pip install positional_encodings==6.0.1
pip install easy_torch==1.2.12

# 数据处理
pip install pandas tables

# TVM (可选，如果安装失败可以跳过)
pip install tvm==1.0.0
```

## 常见问题

### 问题 1: torch 安装失败

如果 torch==1.13.1 安装失败，可以尝试：

```bash
# 使用 PyTorch 官方源
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

或者安装最新版本：

```bash
pip install torch
```

### 问题 2: tvm 安装失败

TVM 不是必需的，如果安装失败可以跳过：

```bash
# 从 requirements.txt 中移除 tvm 后再安装
grep -v "tvm" requirements.txt > requirements_no_tvm.txt
pip install -r requirements_no_tvm.txt
```

### 问题 3: numpy 版本冲突

如果遇到 numpy 版本冲突，可以使用更新的版本：

```bash
pip install numpy>=1.21.5
```

### 问题 4: 权限问题

如果遇到权限问题，确保虚拟环境已激活：

```bash
which python
# 应该显示: /home/xiaoxiao/STD-MAE-main/venv_stdmae/bin/python
```

## 安装后的下一步

安装完成后，您可以：

1. **生成数据集**:
   ```bash
   python scripts/data_preparation/METR-LA/generate_training_data.py
   ```

2. **验证修改**:
   ```bash
   python verify_modifications.py
   ```

3. **运行训练**:
   ```bash
   python stdmae/run.py --cfg='stdmae/STDMAE_METR-LA.py' --gpus='0'
   ```

## 快速检查清单

- [ ] 虚拟环境已激活
- [ ] requirements.txt 中的包已安装
- [ ] pandas 和 tables 已安装
- [ ] 所有导入测试通过
- [ ] 数据集已生成

## 获取帮助

如果遇到问题，请检查：
1. Python 版本是否正确（建议 3.9+）
2. 虚拟环境是否正确激活
3. pip 是否为最新版本：`pip install --upgrade pip`
4. 是否有足够的磁盘空间

---

**创建时间**: 2026-01-17
**虚拟环境路径**: `/home/xiaoxiao/STD-MAE-main/venv_stdmae`
