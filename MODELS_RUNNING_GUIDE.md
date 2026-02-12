# Traffic Prediction Models - Running Guide

This document provides running commands for all integrated traffic prediction models in this repository.

---

## Table of Contents

1. [TrafficFM (Core Method)](#trafficfm-core-method)
2. [HimNet](#himnet)
3. [DiffSTG](#diffstg)
4. [STG4Traffic (DCRNN, DGCRN, GWNET, STGCN)](#stg4traffic)
5. [STD-MAE](#std-mae)
6. [Time-Series-Library (FlowGNN_style scripts)](#time-series-library)

---

## TrafficFM (Core Method)

### Requirements
- Python >= 3.7
- PyTorch >= 1.8
- numpy
- Supports distributed training (DDP)

### METR-LA Dataset

```bash
cd methods/TrafficFM
python train_metr.py \
    --model default \
    --devices 0 \
    --data ../../data/processed/TrafficFM/METR-LA-144-3feat \
    --adjdata ../../data/processed/TrafficFM/sensor_graph/adj_mx.pkl \
    --batch_size 16 \
    --epochs 15
```

### PEMS-BAY Dataset

```bash
cd methods/TrafficFM
python train_pems.py \
    --model default \
    --devices 0 \
    --data ../../data/processed/TrafficFM/PEMS-BAY-144-3feat-row \
    --adjdata ../../data/processed/TrafficFM/sensor_graph/adj_mx_bay.pkl \
    --batch_size 4 \
    --epochs 10
```

### Key Parameters
- `--model`: Model variant (default, ablation1, ablation2)
- `--devices`: GPU device ID(s), comma-separated for multi-GPU (e.g., "0,1")
- `--data`: Data directory path
- `--adjdata`: Adjacency matrix file path
- `--batch_size`: Batch size
- `--epochs`: Number of training epochs
- `--seq_length`: Input sequence length (default: 144)
- `--num_nodes`: Number of nodes (METR-LA: 207, PEMS-BAY: 325)

---

## HimNet

### Requirements
- PyTorch >= 1.8
- numpy
- matplotlib
- pyyaml

### METR-LA Dataset

```bash
cd methods/baselines/HimNet
python scripts/train.py -d METRLA -g 0
```

### PEMS-BAY Dataset

```bash
cd methods/baselines/HimNet
python scripts/train.py -d PEMSBAY -g 0
```

### Key Parameters
- `-d, --dataset`: Dataset name (METRLA, PEMSBAY)
- `-g, --gpu_num`: GPU number (default: 0)
- `-c, --compile`: Enable model compilation (PyTorch 2.0+)
- `--seed`: Random seed (default: 0)
- `--cpus`: Number of CPU threads (default: 1)

---

## DiffSTG

### Requirements
- PyTorch >= 1.8
- numpy
- easydict
- tqdm
- tensorboard (optional)

### METR-LA Dataset (Single GPU)

```bash
cd methods/baselines/DiffSTG
python train.py --data metr-la --gpu_ids 0 --epochs 100

# Or using script
bash run_metr_la_single_gpu.sh
```

### PEMS-BAY Dataset (Single GPU)

```bash
cd methods/baselines/DiffSTG
python train.py --data pems-bay --gpu_ids 0 --epochs 100

# Or using script
bash run_pems_bay_single_gpu.sh
```

### Multi-GPU Training

```bash
cd methods/baselines/DiffSTG
python train.py --data metr-la --use_multi_gpu True --gpu_ids 0,1 --epochs 100
```

### Memory-Optimized Training

```bash
cd methods/baselines/DiffSTG
bash run_metr_la_memory_optimized.sh
bash run_pems_bay_memory_optimized.sh
```

### Key Parameters
- `--data`: Dataset name (metr-la, pems-bay, PEMS08, AIR_GZ)
- `--gpu_ids`: GPU IDs, comma-separated (e.g., "0,1")
- `--use_multi_gpu`: Enable multi-GPU (True/False)
- `--epochs`: Number of epochs (default: 100)
- `--batch_size`: Batch size (default: 8)
- `--test_batch_size`: Test batch size (default: 1)
- `--lr`: Learning rate (default: 0.002)
- `--hidden_size`: Hidden layer size (default: 32)
- `--N`: Diffusion steps (default: 200)
- `--sample_steps`: Sampling steps (default: 200)
- `--T_h`: Historical time steps (default: 144)
- `--n_samples`: Number of samples (default: 8)

---

## STG4Traffic

STG4Traffic contains 4 classic spatiotemporal graph neural network models sharing a unified library.

### Requirements
- PyTorch >= 1.8
- numpy
- scipy

### DCRNN

**METR-LA:**
```bash
cd methods/baselines/STG4Traffic/DCRNN
# Edit DCRNN_Config.py: set DATASET = 'METRLA'
python DCRNN_Main.py
```

**PEMS-BAY:**
```bash
cd methods/baselines/STG4Traffic/DCRNN
# Edit DCRNN_Config.py: set DATASET = 'PEMSBAY'
python DCRNN_Main.py
```

### DGCRN

**METR-LA:**
```bash
cd methods/baselines/STG4Traffic/DGCRN
# Edit DGCRN_Config.py: set DATASET = 'METRLA'
python DGCRN_Main.py
```

**PEMS-BAY:**
```bash
cd methods/baselines/STG4Traffic/DGCRN
# Edit DGCRN_Config.py: set DATASET = 'PEMSBAY'
python DGCRN_Main_PEMSBAY.py
```

### GWNET

**METR-LA:**
```bash
cd methods/baselines/STG4Traffic/GWNET
# Edit GWNET_Config.py: set DATASET = 'METRLA'
python GWNET_Main.py
```

**PEMS-BAY:**
```bash
cd methods/baselines/STG4Traffic/GWNET
# Edit GWNET_Config.py: set DATASET = 'PEMSBAY'
python GWNET_Main.py
```

### STGCN

**METR-LA:**
```bash
cd methods/baselines/STG4Traffic/STGCN
# Edit STGCN_Config.py: set DATASET = 'METRLA'
python STGCN_Main.py
```

**PEMS-BAY:**
```bash
cd methods/baselines/STG4Traffic/STGCN
# Edit STGCN_Config.py: set DATASET = 'PEMSBAY'
python STGCN_Main.py
```

### Configuration
- Edit corresponding `Config.py` file
- Set `MODE`: 'train' or 'test'
- Set `DEVICE`: GPU device (e.g., 'cuda:0')
- Set `DATASET`: 'METRLA' or 'PEMSBAY'
- Set `DEBUG`: Debug mode

---

## STD-MAE

### Requirements
- Python 3.9+ (recommended: 3.9 or 3.13)
- PyTorch 1.13.0+
- BasicTS (included)
- EasyTorch
- See `requirements.txt` or `requirements_py313.txt`

**Recommendation:** Use a separate virtual environment to avoid dependency conflicts.

### METR-LA Dataset

```bash
cd methods/baselines/STD-MAE
python stdmae/run.py --cfg stdmae/STDMAE_METR-LA.py --gpus 0
```

### PEMS-BAY Dataset

**Quick Run (144->144):**
```bash
cd methods/baselines/STD-MAE
bash run_pemsbay_144_quick.sh
```

**Full Training:**
```bash
cd methods/baselines/STD-MAE
bash run_pemsbay_144.sh
```

**Using Python:**
```bash
cd methods/baselines/STD-MAE
python stdmae/run.py --cfg stdmae/STDMAE_PEMS-BAY_144.py --gpus 0
```

### Other Datasets

```bash
cd methods/baselines/STD-MAE

# PEMS03
python stdmae/run.py --cfg stdmae/STDMAE_PEMS03.py --gpus 0

# PEMS04
python stdmae/run.py --cfg stdmae/STDMAE_PEMS04.py --gpus 0

# PEMS07
python stdmae/run.py --cfg stdmae/STDMAE_PEMS07.py --gpus 0

# PEMS08
python stdmae/run.py --cfg stdmae/STDMAE_PEMS08.py --gpus 0
```

### Key Parameters
- `--cfg`: Configuration file path (required)
- `--gpus`: GPU device ID(s), comma-separated (e.g., "0,1")

### Installation Guides
- `INSTALL_GUIDE.md` - Basic installation guide
- `INSTALL_QUICK_REF.md` - Quick reference
- `PYTHON313_INSTALL_GUIDE.md` - Python 3.13 installation
- `DEPENDENCY_CONFLICT_GUIDE.md` - Dependency conflict resolution
- `FIX_NUMPY_SCIPY_GUIDE.md` - NumPy/SciPy issues

---

## Time-Series-Library

Time-Series-Library contains 8 time series forecasting models with FlowGNN-style data preprocessing.

### Requirements
- torch >= 1.9.0
- numpy
- pandas
- matplotlib
- scikit-learn

### Installation

```bash
cd methods/baselines/Time-Series-Library
pip install -r requirements.txt
```

### Available Models

1. **Autoformer** - Autoformer: Decomposition Transformers
2. **FEDformer** - Frequency Enhanced Decomposed Transformer
3. **Mamba** - Mamba: Linear-Time Sequence Modeling
4. **PatchTST** - Patch Time Series Transformer
5. **TimesNet** - TimesNet: Temporal 2D-Variation Modeling
6. **DLinear** - DLinear: Decomposition Linear
7. **Informer** - Informer: Beyond Efficient Transformer
8. **Transformer** - Vanilla Transformer

### METR-LA Dataset

```bash
cd methods/baselines/Time-Series-Library/scripts/long_term_forecast/metrla_script

# Autoformer
./Autoformer_FlowGNN_style.sh

# FEDformer
./FEDformer_FlowGNN_style.sh

# Mamba
./Mamba_FlowGNN_style.sh

# PatchTST
./PatchTST_FlowGNN_style.sh

# TimesNet
./TimesNet_FlowGNN_style.sh

# DLinear
./Dlinear_FlowGNN_style.sh

# Informer
./Informer_FlowGNN_style.sh

# Transformer
./Transformer_FlowGNN_style.sh
```

### PEMS-BAY Dataset

```bash
cd methods/baselines/Time-Series-Library/scripts/long_term_forecast/pemsbay_script

# Autoformer
./Autoformer_FlowGNN_style.sh

# FEDformer
./FEDformer_FlowGNN_style.sh

# Mamba
./Mamba_FlowGNN_style.sh

# PatchTST
./PatchTST_FlowGNN_style.sh

# TimesNet
./TimesNet_FlowGNN_style.sh

# DLinear
./Dlinear_FlowGNN_style.sh

# Informer
./Informer_FlowGNN_style.sh

# Transformer
./Transformer_FlowGNN_style.sh
```

### FlowGNN_style Features

These scripts use FlowGNN-style data preprocessing:
- `--scale_flow_only`: Only standardize flow values
- `--no_overlap`: No overlap between input and output sequences
- Predict 144 steps ahead from 144 input steps (144->144)

---

## Dataset Information

### METR-LA
- **Nodes**: 207
- **Time Range**: 2012/03/01 - 2012/06/30
- **Sampling Interval**: 5 minutes
- **Feature**: Traffic speed

### PEMS-BAY
- **Nodes**: 325
- **Time Range**: 2017/01/01 - 2017/06/30
- **Sampling Interval**: 5 minutes
- **Feature**: Traffic speed

---

## General Notes

### GPU Configuration

Set GPU device before running:
```bash
export CUDA_VISIBLE_DEVICES=0
# Or for multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
```

### Memory Management

- PEMS-BAY has more nodes (325) than METR-LA (207)
- Reduce `batch_size` if encountering out-of-memory errors
- Use memory-optimized scripts when available

### Virtual Environments

Recommended to use virtual environments:

```bash
# Using venv
python3 -m venv venv_traffic
source venv_traffic/bin/activate

# Using conda
conda create -n traffic python=3.9
conda activate traffic
```

### Data Formats

Different methods use different data formats:
- **TrafficFM, HimNet, STG4Traffic**: npz, pkl
- **DiffSTG**: npy
- **STD-MAE**: pkl
- **Time-Series-Library**: csv

All data files are located in `data/processed/[METHOD_NAME]/`

---

## Quick Summary Table

| Method | Type | Datasets | Data Format | Framework |
|--------|------|----------|-------------|----------|
| TrafficFM | Core | METR-LA, PEMS-BAY | npz, pkl | Standalone |
| HimNet | Baseline | METR-LA, PEMS-BAY | npz, pkl | Standalone |
| DiffSTG | Baseline | METR-LA, PEMS-BAY, PEMS08 | npy | Standalone |
| DCRNN | Baseline | METR-LA, PEMS-BAY | npz, pkl | STG4Traffic |
| DGCRN | Baseline | METR-LA, PEMS-BAY | npz, pkl | STG4Traffic |
| GWNET | Baseline | METR-LA, PEMS-BAY | npz, pkl | STG4Traffic |
| STGCN | Baseline | METR-LA, PEMS-BAY | npz, pkl | STG4Traffic |
| STD-MAE | Baseline | METR-LA, PEMS-BAY, PEMS03-08 | pkl | BasicTS |
| Time-Series-Library (8 models) | Baseline Scripts | METR-LA, PEMS-BAY | csv | Standalone |

---

## References

For detailed information about each method, please refer to:
- `INTEGRATION_LOG.md` - Detailed integration records
- `BASELINE_INTEGRATION_PROTOCOL.md` - Integration protocol
- Individual method README files in their respective directories

---

**Total Integrated Methods**: 15+ models across 7 method families

**Supported Datasets**: METR-LA, PEMS-BAY, PEMS03, PEMS04, PEMS07, PEMS08
