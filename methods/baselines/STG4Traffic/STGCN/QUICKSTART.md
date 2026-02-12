# STGCN NPY Generation - Quick Start Guide

## What Was Created

Three scripts have been created to generate STGCN prediction results in NPY format:

1. **`generate_predictions_npy.py`** - Main script that generates predictions
2. **`run_generate_predictions.py`** - Python wrapper to run both datasets easily
3. **`generate_all_predictions.sh`** - Bash script alternative
4. **`README_NPY_GENERATION.md`** - Detailed documentation

## Quick Start (Recommended)

Run this single command to generate predictions for both PEMSBAY and METRLA:

```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN
python3 run_generate_predictions.py
```

This will:
- Automatically activate the stg4t conda environment
- Load the trained STGCN models
- Generate predictions for PEMSBAY (325 nodes, 144 time steps)
- Generate predictions for METRLA (207 nodes, 144 time steps)
- Save 4 NPY files to `./predictions_npy/` directory

## Output Files

After running, you will get these files in `./predictions_npy/`:

1. **`stgcn_pemsbay_predictions.npy`** - STGCN predictions for PEMSBAY
   - Shape: (144, num_samples, 325)

2. **`pemsbay_ground_truth.npy`** - Ground truth for PEMSBAY
   - Shape: (144, num_samples, 325)

3. **`stgcn_metrla_predictions.npy`** - STGCN predictions for METRLA
   - Shape: (144, num_samples, 207)

4. **`metrla_ground_truth.npy`** - Ground truth for METRLA
   - Shape: (144, num_samples, 207)

## Format Specification

All NPY files follow the required format:
- **Shape**: `(seq_length, num_samples, num_nodes)`
  - `seq_length`: 144 (prediction horizon)
  - `num_samples`: Number of test samples
  - `num_nodes`: 325 for PEMSBAY, 207 for METRLA
- **Data type**: float32 or float64
- **Values**: De-normalized real values (actual traffic speeds)

## Alternative Usage

### Run single dataset only:

```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate stg4t

# PEMSBAY only
python generate_predictions_npy.py --dataset PEMSBAY --device cuda:0

# METRLA only
python generate_predictions_npy.py --dataset METRLA --device cuda:0
```

### Specify custom model path:

```bash
python generate_predictions_npy.py \
    --dataset PEMSBAY \
    --model_path /path/to/model.pth \
    --device cuda:0
```

## Trained Models Used

The scripts automatically detect and use these trained models:
- **PEMSBAY**: `../log/STGCN/PEMSBAY/20251229012414/PEMSBAY_STGCN_best_model.pth`
- **METRLA**: `../log/STGCN/METRLA/20251229023643/METRLA_STGCN_best_model.pth`

## Loading Generated NPY Files

```python
import numpy as np

# Load predictions
pemsbay_pred = np.load('predictions_npy/stgcn_pemsbay_predictions.npy')
pemsbay_gt = np.load('predictions_npy/pemsbay_ground_truth.npy')

metrla_pred = np.load('predictions_npy/stgcn_metrla_predictions.npy')
metrla_gt = np.load('predictions_npy/metrla_ground_truth.npy')

# Check shapes
print(f"PEMSBAY predictions: {pemsbay_pred.shape}")  # (144, num_samples, 325)
print(f"METRLA predictions: {metrla_pred.shape}")    # (144, num_samples, 207)
```

## Troubleshooting

### CUDA out of memory
Reduce batch size:
```bash
python generate_predictions_npy.py --dataset PEMSBAY --batch_size 32
```

Or use CPU:
```bash
python generate_predictions_npy.py --dataset PEMSBAY --device cpu
```

### Model not found
Manually specify model path:
```bash
python generate_predictions_npy.py \
    --dataset PEMSBAY \
    --model_path /full/path/to/model.pth
```

## More Information

See `README_NPY_GENERATION.md` for detailed documentation including:
- Complete command line arguments
- Output format verification
- Example outputs
- Advanced usage scenarios
