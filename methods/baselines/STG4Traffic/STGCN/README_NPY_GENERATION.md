# STGCN Prediction NPY Generation

This directory contains scripts to generate STGCN prediction results in NPY format according to the specification.

## Output Format

All NPY files follow this format:
- **Shape**: `(seq_length, num_samples, num_nodes)`
  - `seq_length`: Prediction horizon (144 time steps)
  - `num_samples`: Number of test samples
  - `num_nodes`: Number of nodes (325 for PEMSBAY, 207 for METRLA)
- **Data type**: `float32` or `float64`
- **Values**: De-normalized real values (actual traffic speeds)

## Files Generated

For each dataset, two files are generated:

### PEMSBAY
- `stgcn_pemsbay_predictions.npy` - STGCN model predictions
- `pemsbay_ground_truth.npy` - Ground truth values

### METRLA
- `stgcn_metrla_predictions.npy` - STGCN model predictions
- `metrla_ground_truth.npy` - Ground truth values

## Usage

### Option 1: Generate predictions for both datasets (Recommended)

**Method A: Using Python wrapper (Easiest)**
```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN
python3 run_generate_predictions.py
```

**Method B: Using bash script**
```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN
./generate_all_predictions.sh
```

Both methods will automatically:
1. Activate the correct conda environment (stg4t)
2. Generate predictions for PEMSBAY
3. Generate predictions for METRLA
4. Save all files to `./predictions_npy/` directory

### Option 2: Generate predictions for a single dataset

First activate the conda environment:
```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate stg4t
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN
```

Then run for specific dataset:
```bash
# For PEMSBAY
python generate_predictions_npy.py --dataset PEMSBAY --device cuda:0

# For METRLA
python generate_predictions_npy.py --dataset METRLA --device cuda:0
```

### Option 3: Specify custom model path

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate stg4t
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN

python generate_predictions_npy.py \
    --dataset PEMSBAY \
    --model_path /path/to/your/model.pth \
    --device cuda:0 \
    --output_dir ./my_predictions
```

## Command Line Arguments

- `--dataset`: Dataset to use (METRLA or PEMSBAY) [required]
- `--model_path`: Path to trained model checkpoint [optional, auto-detects if not specified]
- `--device`: Device to use (cuda:0, cuda:1, or cpu) [default: cuda:0]
- `--batch_size`: Batch size for testing [default: 64]
- `--output_dir`: Directory to save NPY files [default: ./predictions_npy]

## Model Path Auto-Detection

If `--model_path` is not specified, the script will automatically search for the best model:

- **PEMSBAY**: `../log/STGCN/PEMSBAY/*/PEMSBAY_STGCN_best_model.pth`
- **METRLA**: `../log/STGCN/METRLA/*/METRLA_STGCN_best_model.pth`

The script will use the most recently modified model file.

**Current trained models:**
- PEMSBAY: `/home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/log/STGCN/PEMSBAY/20251229012414/PEMSBAY_STGCN_best_model.pth`
- METRLA: `/home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/log/STGCN/METRLA/20251229023643/METRLA_STGCN_best_model.pth`

## Verification

After generation, the script will display:
- Shape of predictions and ground truth
- Data type
- File size
- Value range
- Overall metrics (MAE, RMSE, MAPE)

## Example Output

```
================================================================================
STGCN Prediction NPY Generator
================================================================================
Dataset: PEMSBAY
Device: cuda:0
Num Nodes: 325
Window: 144
Horizon: 144
Output Dir: ./predictions_npy
================================================================================

Loading data...
Data loaded successfully!

Generating model...
Model generated successfully!

Using model: ../log/STGCN/PEMSBAY/20240115_123456/PEMSBAY_STGCN_best_model.pth

Generating predictions...
  Processed time step 12/144
  Processed time step 24/144
  ...
  Processed time step 144/144

Predictions shape: (144, 5209, 325)
Ground truth shape: (144, 5209, 325)
Data type: float32

Format verification:
  seq_length (horizon): 144
  num_samples: 5209
  num_nodes: 325

Overall Metrics:
  MAE:  1.2345
  RMSE: 2.3456
  MAPE: 3.45%

Saving NPY files...

Saved predictions to: ./predictions_npy/stgcn_pemsbay_predictions.npy
Saved ground truth to: ./predictions_npy/pemsbay_ground_truth.npy

File information:
  Predictions file: stgcn_pemsbay_predictions.npy
    - Shape: (144, 5209, 325)
    - Dtype: float32
    - Size: 965.23 MB
    - Value range: [0.00, 75.50]

  Ground truth file: pemsbay_ground_truth.npy
    - Shape: (144, 5209, 325)
    - Dtype: float32
    - Size: 965.23 MB
    - Value range: [0.00, 75.50]

================================================================================
NPY generation completed successfully!
Files saved to: ./predictions_npy
================================================================================
```

## Loading NPY Files

To load and use the generated NPY files:

```python
import numpy as np

# Load predictions
predictions = np.load('predictions_npy/stgcn_pemsbay_predictions.npy')
ground_truth = np.load('predictions_npy/pemsbay_ground_truth.npy')

# Check shape
print(f"Predictions shape: {predictions.shape}")  # (144, num_samples, 325)

# Access specific time step
time_step = 0
pred_t0 = predictions[time_step]  # Shape: (num_samples, 325)

# Access specific sample
sample_idx = 0
pred_sample = predictions[:, sample_idx, :]  # Shape: (144, 325)

# Access specific node
node_idx = 0
pred_node = predictions[:, :, node_idx]  # Shape: (144, num_samples)
```

## Troubleshooting

### Model not found
If you get "Model file not found" error:
1. Check if the model has been trained
2. Verify the model path pattern in the script
3. Manually specify the model path using `--model_path`

### CUDA out of memory
If you encounter CUDA memory errors:
1. Reduce batch size: `--batch_size 32`
2. Use CPU: `--device cpu`

### Import errors
Make sure you're running from the STGCN directory:
```bash
cd /home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN
```

## Notes

- The script follows the same model loading approach as `test_and_plot.py`
- Predictions are de-normalized to real values using the scaler
- The output format matches the specification: `(seq_length, num_samples, num_nodes)`
- All values are in the original scale (e.g., actual traffic speeds in mph or km/h)
