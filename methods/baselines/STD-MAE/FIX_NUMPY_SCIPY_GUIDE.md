# NumPy/SciPy 版本不兼容问题修复指南

## 错误信息

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

## 问题原因

这是一个**二进制不兼容**问题，发生在：
- NumPy 版本太新（2.2.5）
- SciPy 是用旧版本 NumPy 编译的
- SciPy 期望 NumPy <1.23.0，但实际安装了 2.2.5

## 🚀 快速修复

### 方法 1: 使用修复脚本（最简单）

```bash
conda activate stdmae_new
cd /home/xiaoxiao/STD-MAE-main
bash fix_numpy_scipy.sh
```

### 方法 2: 手动修复（推荐的版本组合）

```bash
conda activate stdmae_new

# 卸载不兼容的版本
pip uninstall numpy scipy -y

# 安装兼容的版本组合
pip install numpy==1.21.6 scipy==1.7.3

# 验证
python -c "import scipy.sparse; print('✓ 修复成功')"
```

### 方法 3: 使用最新兼容版本

```bash
conda activate stdmae_new

# 卸载旧版本
pip uninstall numpy scipy -y

# 安装最新兼容版本
pip install "numpy>=1.21.0,<1.23.0"
pip install scipy

# 验证
python -c "import scipy.sparse; print('✓ 修复成功')"
```

## 📋 推荐的版本组合

### 组合 1: 稳定版（与原始 requirements.txt 一致）

```bash
numpy==1.21.6
scipy==1.7.3
torch==1.13.1
```

**优点**:
- 与论文环境完全一致
- 最稳定
- 已验证可用

**缺点**:
- 缺少新特性
- 可能有已知 bug

### 组合 2: 兼容版（推荐）

```bash
numpy>=1.21.0,<1.23.0
scipy>=1.7.0,<1.10.0
torch>=1.13.0
```

**优点**:
- 包含 bug 修复
- 性能改进
- 仍然兼容

**缺点**:
- 可能有轻微的行为差异

### 组合 3: 最新版（需要 Python 3.9+）

```bash
numpy>=1.26.0
scipy>=1.11.0
torch>=2.0.0
```

**优点**:
- 最新特性
- 最佳性能
- 安全更新

**缺点**:
- 需要 Python 3.9+
- 可能有兼容性问题

## 🔍 验证修复

运行以下命令验证修复是否成功：

```bash
conda activate stdmae_new

# 测试 1: 检查版本
python << 'EOF'
import numpy
import scipy
print(f"NumPy: {numpy.__version__}")
print(f"SciPy: {scipy.__version__}")
EOF

# 测试 2: 导入 scipy.sparse
python -c "import scipy.sparse; print('✓ scipy.sparse 导入成功')"

# 测试 3: 运行数据生成脚本
python scripts/data_preparation/METR-LA/generate_training_data.py
```

## 🐛 其他可能的问题

### 问题 1: 仍然报错

**原因**: 可能有缓存的编译文件

**解决**:
```bash
# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# 重新安装
pip uninstall numpy scipy -y
pip install --no-cache-dir numpy==1.21.6 scipy==1.7.3
```

### 问题 2: pip 安装失败

**原因**: 网络问题或源问题

**解决**:
```bash
# 使用清华镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.21.6 scipy==1.7.3
```

### 问题 3: conda 和 pip 混用导致冲突

**原因**: conda 和 pip 安装的包冲突

**解决**:
```bash
# 优先使用 conda 安装
conda install numpy=1.21.6 scipy=1.7.3 -y

# 或者创建纯 pip 环境
conda create -n stdmae_pip python=3.10 -y
conda activate stdmae_pip
pip install numpy==1.21.6 scipy==1.7.3
```

## 📊 版本兼容性表

| NumPy 版本 | SciPy 版本 | Python 版本 | 状态 |
|-----------|-----------|------------|------|
| 1.21.x | 1.7.x | 3.7-3.10 | ✅ 推荐 |
| 1.22.x | 1.8.x | 3.8-3.10 | ✅ 可用 |
| 1.23.x | 1.9.x | 3.8-3.11 | ✅ 可用 |
| 1.24.x | 1.10.x | 3.8-3.11 | ✅ 可用 |
| 1.26.x | 1.11.x | 3.9-3.12 | ✅ 可用 |
| 2.x | 1.7.x | 任意 | ❌ 不兼容 |
| 2.x | 1.13.x+ | 3.9+ | ✅ 可用 |

## 🎯 针对您的环境

您当前使用的是 **conda 环境 stdmae_new (Python 3.10)**，推荐配置：

```bash
conda activate stdmae_new

# 方案 A: 稳定版（推荐）
pip uninstall numpy scipy -y
pip install numpy==1.21.6 scipy==1.7.3

# 方案 B: 较新版本
pip uninstall numpy scipy -y
pip install numpy==1.23.5 scipy==1.9.3

# 方案 C: 使用 conda（更可靠）
conda install numpy=1.21.6 scipy=1.7.3 -y
```

## 📝 完整的依赖安装

如果要重新安装所有依赖：

```bash
conda activate stdmae_new

# 卸载可能冲突的包
pip uninstall numpy scipy torch -y

# 安装核心依赖（按顺序）
pip install numpy==1.21.6
pip install scipy==1.7.3
pip install torch==1.13.1

# 安装其他依赖
pip install pandas tables easydict scikit-learn==1.0.2
pip install setproctitle sympy timm==0.6.11
pip install torch-summary positional-encodings

# 安装 EasyTorch
pip install easy_torch==1.2.12
```

## ✅ 修复后的下一步

修复成功后，继续执行：

```bash
# 1. 生成数据集
python scripts/data_preparation/METR-LA/generate_training_data.py

# 2. 验证修改
python verify_modifications.py

# 3. 测试导入
python -c "from stdmae.stdmae_arch import STDMAE; print('✓ 模型导入成功')"
```

---

**更新时间**: 2026-01-17
**环境**: conda stdmae_new (Python 3.10)
**推荐版本**: numpy==1.21.6, scipy==1.7.3
