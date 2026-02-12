import numpy as np

# 加载PEMSBAY测试数据
pemsbay_test = np.load('../data/PEMS-BAY/processed/test.npz')
print('PEMSBAY test.npz keys:', list(pemsbay_test.keys()))
print('PEMSBAY x_test shape:', pemsbay_test['x'].shape)
print('PEMSBAY y_test shape:', pemsbay_test['y'].shape)
print()

# 加载METRLA测试数据
metrla_test = np.load('../data/METR-LA/processed/test.npz')
print('METRLA test.npz keys:', list(metrla_test.keys()))
print('METRLA x_test shape:', metrla_test['x'].shape)
print('METRLA y_test shape:', metrla_test['y'].shape)
