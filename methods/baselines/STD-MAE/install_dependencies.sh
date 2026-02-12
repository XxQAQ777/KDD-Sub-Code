#!/bin/bash

# 激活虚拟环境
source venv_stdmae/bin/activate

echo "=========================================="
echo "安装 requirements.txt 中的依赖..."
echo "=========================================="

# 安装 requirements.txt 中的依赖
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "安装额外依赖 (pandas, tables)..."
echo "=========================================="

# 安装额外需要的依赖
pip install pandas tables

echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="

# 验证关键包是否安装成功
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')"
python -c "import pandas; print(f'✓ Pandas {pandas.__version__}')"
python -c "import tables; print(f'✓ Tables {tables.__version__}')"
python -c "import easydict; print('✓ EasyDict installed')"
python -c "import scipy; print(f'✓ SciPy {scipy.__version__}')"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
