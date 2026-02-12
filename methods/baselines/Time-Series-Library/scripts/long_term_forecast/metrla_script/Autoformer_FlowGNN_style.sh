#!/usr/bin/env bash
set -euo pipefail

# TrafficFM-main integration wrapper for Time-Series-Library FlowGNN_style (METR-LA)
# - Uses unified data dir under TrafficFM-main/data/processed/Time-Series-Library/
# - Locates Time-Series-Library code (run.py) without hardcoded absolute paths

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAFFICFM_ROOT="$(cd "$SCRIPT_DIR/../../../../../../" && pwd)"
TSL_DATA_ROOT="$TRAFFICFM_ROOT/data/processed/Time-Series-Library"

# Candidate locations for the Time-Series-Library codebase (must contain run.py)
CANDIDATES=(
  "$TRAFFICFM_ROOT/extern/Time-Series-Library-main"
  "$TRAFFICFM_ROOT/third_party/Time-Series-Library-main"
  "$TRAFFICFM_ROOT/methods/baselines/Time-Series-Library"
)
TSL_ROOT=""
for d in "${CANDIDATES[@]}"; do
  if [ -f "$d/run.py" ]; then TSL_ROOT="$d"; break; fi
done
if [ -z "$TSL_ROOT" ]; then
  echo "Error: Time-Series-Library run.py not found. Place the code under one of:" 1>&2
  printf '  - %s\n' "${CANDIDATES[@]}" 1>&2
  exit 1
fi

# Original script body (paths adjusted)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=Autoformer

# FlowGNN-style preprocessing:
# - Only standardize flow values (--scale_flow_only)
# - No overlap between input and output (--no_overlap)
# - 从144步预测之后144步

python -u "$TSL_ROOT/run.py" \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$TSL_DATA_ROOT/" \
  --data_path metr-la.csv \
  --model_id METR_LA_144_144_FlowGNN_style \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 144 \
  --label_len 72 \
  --pred_len 144 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 207 \
  --dec_in 207 \
  --c_out 207 \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 2048 \
  --moving_avg 25 \
  --dropout 0.1 \
  --des 'Exp' \
  --batch_size 32 \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.0001 \
  --use_amp \
  --itr 1 \
  --inverse \
  --scale_flow_only \
  --no_overlap
