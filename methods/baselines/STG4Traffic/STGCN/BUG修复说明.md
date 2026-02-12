# Bug修复说明

## 修复的问题

### 1. `source: not found` 错误
**原因**: Python的 `subprocess.run()` 默认使用 `/bin/sh`，而 `/bin/sh` 不支持 `source` 命令。

**解决方案**: 在 `subprocess.run()` 中明确指定使用 `/bin/bash`：
```python
result = subprocess.run(cmd, shell=True, executable='/bin/bash', ...)
```

### 2. 不支持指定GPU设备
**原因**: 原脚本硬编码使用 `cuda:0`，无法选择其他GPU。

**解决方案**: 添加命令行参数支持：
```python
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to use: cuda:0, cuda:1, or cpu')
```

## 现在可以使用的命令

### 使用CUDA:1（你的需求）
```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN
python3 run_generate_predictions.py --device cuda:1
```

### 使用CUDA:0（默认）
```bash
python3 run_generate_predictions.py
```

### 使用CPU
```bash
python3 run_generate_predictions.py --device cpu
```

### 自定义批次大小（如果GPU内存不足）
```bash
python3 run_generate_predictions.py --device cuda:1 --batch_size 32
```

## 修改的文件

1. **`run_generate_predictions.py`**
   - 添加了 `argparse` 支持
   - 添加了 `--device` 参数
   - 添加了 `--batch_size` 参数
   - 添加了 `--output_dir` 参数
   - 修复了 `subprocess.run()` 使用 `/bin/bash`

## 测试命令

现在你可以运行：
```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN
python3 run_generate_predictions.py --device cuda:1
```

这将：
1. 自动激活 `stg4t` conda环境
2. 使用 `cuda:1` 设备
3. 生成PEMSBAY的预测结果
4. 生成METRLA的预测结果
5. 保存4个NPY文件到 `./predictions_npy/` 目录

## 输出文件

- `stgcn_pemsbay_predictions.npy` - 形状: (144, num_samples, 325)
- `pemsbay_ground_truth.npy` - 形状: (144, num_samples, 325)
- `stgcn_metrla_predictions.npy` - 形状: (144, num_samples, 207)
- `metrla_ground_truth.npy` - 形状: (144, num_samples, 207)

所有文件都符合规格要求：`(seq_length, num_samples, num_nodes)`
