# 依赖版本冲突完整解决方案

## 问题分析

您遇到了一个**依赖链冲突**：

```
SciPy 1.7.3 → 需要 NumPy < 1.23.0
Pandas (新版) → 需要 NumPy >= 1.22.4
冲突！
```

## 🚀 快速修复（推荐）

### 一键修复

```bash
conda activate stdmae_new
cd /home/xiaoxiao/STD-MAE-main
bash quick_fix_all.sh
```

这将安装最稳定的版本组合。

## 📋 兼容的版本组合

### 组合 1: 稳定版（推荐）⭐

```bash
numpy==1.21.6
scipy==1.7.3
pandas==1.3.5
tables==3.7.0
torch==1.13.1
```

**优点**:
- 完全兼容
- 最稳定
- 与原始项目一致

**安装命令**:
```bash
pip uninstall numpy scipy pandas tables -y
pip install numpy==1.21.6 scipy==1.7.3 pandas==1.3.5 tables==3.7.0
```

### 组合 2: 较新版

```bash
numpy==1.23.5
scipy==1.9.3
pandas==1.5.3
tables==3.8.0
torch==1.13.1
```

**优点**:
- 包含 bug 修复
- 性能改进

**安装命令**:
```bash
pip uninstall numpy scipy pandas tables -y
pip install numpy==1.23.5 scipy==1.9.3 pandas==1.5.3 tables==3.8.0
```

### 组合 3: 最新版（需要 Python 3.9+）

```bash
numpy>=1.26.0
scipy>=1.11.0
pandas>=2.0.0
tables>=3.9.0
torch>=2.0.0
```

**优点**:
- 最新特性
- 最佳性能

**安装命令**:
```bash
pip uninstall numpy scipy pandas tables torch -y
pip install numpy scipy pandas tables torch
```

## 🔧 详细修复步骤

### 方法 1: 使用交互式脚本

```bash
conda activate stdmae_new
cd /home/xiaoxiao/STD-MAE-main
bash fix_all_dependencies.sh
```

脚本会提供两个方案供您选择。

### 方法 2: 手动修复（稳定版）

```bash
conda activate stdmae_new

# 步骤 1: 卸载冲突的包
pip uninstall numpy scipy pandas tables -y

# 步骤 2: 按顺序安装兼容版本
pip install numpy==1.21.6
pip install scipy==1.7.3
pip install pandas==1.3.5
pip install tables==3.7.0

# 步骤 3: 验证
python -c "import numpy, scipy, pandas, tables; print('✓ 所有包已安装')"
python -c "import scipy.sparse; print('✓ scipy.sparse 导入成功')"
```

### 方法 3: 使用 conda（最可靠）

```bash
conda activate stdmae_new

# 使用 conda 安装（自动解决依赖）
conda install numpy=1.21.6 scipy=1.7.3 pandas=1.3.5 pytables=3.7.0 -y
```

## 📊 版本兼容性矩阵

| NumPy | SciPy | Pandas | Tables | Python | 状态 |
|-------|-------|--------|--------|--------|------|
| 1.21.6 | 1.7.3 | 1.3.5 | 3.7.0 | 3.7-3.10 | ✅ 推荐 |
| 1.22.4 | 1.8.1 | 1.4.4 | 3.7.0 | 3.8-3.10 | ✅ 可用 |
| 1.23.5 | 1.9.3 | 1.5.3 | 3.8.0 | 3.8-3.11 | ✅ 可用 |
| 1.24.3 | 1.10.1 | 2.0.3 | 3.8.0 | 3.8-3.11 | ✅ 可用 |
| 1.26.x | 1.11.x | 2.1.x | 3.9.x | 3.9-3.12 | ✅ 可用 |
| 1.21.6 | 1.7.3 | 2.x | 任意 | 任意 | ❌ 冲突 |
| 2.x | 1.7.3 | 任意 | 任意 | 任意 | ❌ 冲突 |

## 🔍 验证修复

运行以下命令验证所有包都正确安装：

```bash
conda activate stdmae_new

# 完整验证脚本
python << 'EOF'
import sys
print(f"Python: {sys.version}")
print()

# 检查核心包
packages = {
    'numpy': 'NumPy',
    'scipy': 'SciPy',
    'pandas': 'Pandas',
    'tables': 'Tables',
    'torch': 'PyTorch',
}

for module, name in packages.items():
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name}: {version}")
    except ImportError as e:
        print(f"✗ {name}: NOT INSTALLED ({e})")

# 测试关键导入
print()
try:
    import scipy.sparse
    print("✓ scipy.sparse 导入成功")
except Exception as e:
    print(f"✗ scipy.sparse 导入失败: {e}")

try:
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2, 3]})
    print("✓ pandas 功能正常")
except Exception as e:
    print(f"✗ pandas 功能异常: {e}")

try:
    import tables
    print("✓ tables (HDF5) 导入成功")
except Exception as e:
    print(f"✗ tables 导入失败: {e}")
EOF
```

## 🐛 常见问题

### Q1: 安装后仍然报错

**解决**: 清理缓存后重新安装

```bash
# 清理 pip 缓存
pip cache purge

# 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# 重新安装
pip uninstall numpy scipy pandas tables -y
pip install --no-cache-dir numpy==1.21.6 scipy==1.7.3 pandas==1.3.5 tables==3.7.0
```

### Q2: pip 和 conda 混用导致问题

**解决**: 统一使用 conda

```bash
# 卸载 pip 安装的包
pip uninstall numpy scipy pandas tables -y

# 使用 conda 安装
conda install numpy=1.21.6 scipy=1.7.3 pandas=1.3.5 pytables=3.7.0 -y
```

### Q3: 网络问题导致安装失败

**解决**: 使用国内镜像源

```bash
# 清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy==1.21.6 scipy==1.7.3 pandas==1.3.5 tables==3.7.0

# 或配置 conda 镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```

### Q4: 某个包安装失败

**解决**: 跳过该包或使用替代版本

```bash
# 如果 tables 安装失败，可以尝试不同版本
pip install tables==3.6.1
# 或
pip install tables==3.8.0
```

## 📝 完整的环境配置

如果要从头开始配置环境：

```bash
# 创建新环境
conda create -n stdmae_clean python=3.10 -y
conda activate stdmae_clean

# 安装核心依赖（按顺序）
pip install numpy==1.21.6
pip install scipy==1.7.3
pip install pandas==1.3.5
pip install tables==3.7.0

# 安装 PyTorch
pip install torch==1.13.1 torchvision torchaudio

# 安装其他依赖
pip install easydict==1.10
pip install scikit-learn==1.0.2
pip install setproctitle==1.3.2
pip install sympy==1.10.1
pip install timm==0.6.11
pip install torch-summary==1.4.5
pip install positional-encodings==6.0.1

# 安装 EasyTorch
pip install easy_torch==1.2.12

# 验证
python -c "import numpy, scipy, pandas, tables, torch; print('✓ 环境配置完成')"
```

## ✅ 修复后的下一步

修复成功后：

```bash
# 1. 测试数据生成
python scripts/data_preparation/METR-LA/generate_training_data.py

# 2. 验证修改
python verify_modifications.py

# 3. 测试模型导入
python -c "from stdmae.stdmae_arch import STDMAE; print('✓ 模型导入成功')"
```

## 🎯 推荐配置总结

对于您的环境（Python 3.10），**强烈推荐使用组合 1（稳定版）**：

```bash
conda activate stdmae_new
bash quick_fix_all.sh
```

这是经过验证的最稳定配置，可以避免所有已知的依赖冲突。

---

**更新时间**: 2026-01-17
**环境**: conda stdmae_new (Python 3.10)
**推荐版本**: numpy==1.21.6, scipy==1.7.3, pandas==1.3.5, tables==3.7.0
