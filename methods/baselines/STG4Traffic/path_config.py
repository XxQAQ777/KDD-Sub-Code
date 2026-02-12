"""STG4Traffic 统一路径配置

此文件为 TrafficFM-main 集成提供统一的数据路径。
所有模型(DCRNN, DGCRN, GWNET, STGCN)都应该使用这个配置。
"""

import os

# 获取当前文件所在目录 (STG4Traffic)
STG4TRAFFIC_ROOT = os.path.dirname(os.path.abspath(__file__))

# 获取 TrafficFM-main 根目录
# STG4Traffic -> baselines -> methods -> TrafficFM-main
TRAFFICFM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(STG4TRAFFIC_ROOT)))

# 数据根目录
DATA_ROOT = os.path.join(TRAFFICFM_ROOT, 'data', 'processed', 'STG4Traffic')

# 数据集路径
METRLA_DIR = os.path.join(DATA_ROOT, 'METR-LA', 'processed')
PEMSBAY_DIR = os.path.join(DATA_ROOT, 'PEMS-BAY', 'processed')

# 邻接矩阵路径
METRLA_GRAPH = os.path.join(METRLA_DIR, 'adj_mx.pkl')
PEMSBAY_GRAPH = os.path.join(PEMSBAY_DIR, 'adj_mx_bay.pkl')

# 日志根目录 (保留在模型内部)
LOG_ROOT = os.path.join(STG4TRAFFIC_ROOT, 'log')

# 打印配置（用于调试）
if __name__ == '__main__':
    print("STG4Traffic Path Configuration:")
    print(f"  STG4TRAFFIC_ROOT: {STG4TRAFFIC_ROOT}")
    print(f"  TRAFFICFM_ROOT: {TRAFFICFM_ROOT}")
    print(f"  DATA_ROOT: {DATA_ROOT}")
    print(f"  METRLA_DIR: {METRLA_DIR}")
    print(f"  PEMSBAY_DIR: {PEMSBAY_DIR}")
    print(f"  METRLA_GRAPH: {METRLA_GRAPH}")
    print(f"  PEMSBAY_GRAPH: {PEMSBAY_GRAPH}")
    print(f"  LOG_ROOT: {LOG_ROOT}")
