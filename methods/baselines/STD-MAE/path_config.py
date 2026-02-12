"""STD-MAE 路径配置

此文件为 TrafficFM-main 集成提供统一的数据路径。
STD-MAE 使用 BasicTS 框架，保持独立的目录结构。
"""

import os

# 获取当前文件所在目录 (STD-MAE)
STDMAE_ROOT = os.path.dirname(os.path.abspath(__file__))

# 获取 TrafficFM-main 根目录
# STD-MAE -> baselines -> methods -> TrafficFM-main
TRAFFICFM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(STDMAE_ROOT)))

# 数据根目录
DATA_ROOT = os.path.join(TRAFFICFM_ROOT, 'data', 'processed', 'STD-MAE')

# 数据集路径
METRLA_DIR = os.path.join(DATA_ROOT, 'METR-LA')
PEMSBAY_DIR = os.path.join(DATA_ROOT, 'PEMS-BAY')
RAW_DATA_DIR = os.path.join(DATA_ROOT, 'raw_data')

# 检查点目录 (保留在模型内部)
CHECKPOINT_DIR = os.path.join(STDMAE_ROOT, 'checkpoints')

# 打印配置（用于调试）
if __name__ == '__main__':
    print("STD-MAE Path Configuration:")
    print(f"  STDMAE_ROOT: {STDMAE_ROOT}")
    print(f"  TRAFFICFM_ROOT: {TRAFFICFM_ROOT}")
    print(f"  DATA_ROOT: {DATA_ROOT}")
    print(f"  METRLA_DIR: {METRLA_DIR}")
    print(f"  PEMSBAY_DIR: {PEMSBAY_DIR}")
    print(f"  RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"  CHECKPOINT_DIR: {CHECKPOINT_DIR}")
