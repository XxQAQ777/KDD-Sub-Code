#!/usr/bin/env bash
set -euo pipefail

# TrafficFM-main integration wrapper (METR-LA / PatchTST)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAFFICFM_ROOT="$(cd "$SCRIPT_DIR/../../../../../../" && pwd)"
TSL_DATA_ROOT="$TRAFFICFM_ROOT/data/processed/Time-Series-Library"
CANDIDATES=("$TRAFFICFM_ROOT/extern/Time-Series-Library-main" "$TRAFFICFM_ROOT/third_party/Time-Series-Library-main" "$TRAFFICFM_ROOT/methods/baselines/Time-Series-Library")
TSL_ROOT=""; for d in "${CANDIDATES[@]}"; do [ -f "$d/run.py" ] && TSL_ROOT="$d" && break; done;
if [ -z "$TSL_ROOT" ]; then echo "Error: Time-Series-Library run.py not found" 1>&2; printf '  - %s\n' "${CANDIDATES[@]}" 1>&2; exit 1; fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=PatchTST

# FlowGNN-style preprocessing
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
  --d_model 128 \
  --n_heads 4 \
  --d_ff 256 \
  --dropout 0.1 \
  --patch_len 16 \
  --des 'Exp' \
  --batch_size 32 \
  --train_epochs 100 \
  --patience 10 \
  --learning_rate 0.0001 \
  --use_amp \
  --itr 1 \
  --inverse \
  --is_training 0 \
  --scale_flow_only \
  --no_overlap
