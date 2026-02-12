#!/usr/bin/env python3
"""
Generate NPY prediction files for HimNet model
Format: (seq_length, num_samples, num_nodes)
"""

import sys
import os
import torch
import numpy as np
import yaml
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.utils import seed_everything, set_cpu_num
from lib.data_prepare import get_dataloaders_from_index_data
from model.HimNet import HimNet
from model.HimNetRunner import HimNetRunner


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate NPY prediction files for HimNet')

    parser.add_argument('--dataset', type=str, default='PEMSBAY',
                        choices=['METRLA', 'PEMSBAY'],
                        help='Dataset to use: METRLA or PEMSBAY')

    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model checkpoint. If not specified, will use default path.')

    parser.add_argument('--output_dir', type=str, default='npy_predictions',
                        help='Directory to save NPY files')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use: cuda:0, cuda:1, or cpu')

    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    parser.add_argument('--cpus', type=int, default=1,
                        help='Number of CPUs to use')

    args = parser.parse_args()
    return args


def load_data(dataset, project_root, cfg):
    """Load dataset"""
    data_path = os.path.join(project_root, "data", dataset)

    trainset_loader, valset_loader, testset_loader, SCALER = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        y_tod=cfg.get("y_time_of_day"),
        y_dow=cfg.get("y_day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=None,
    )

    return trainset_loader, valset_loader, testset_loader, SCALER


def load_model(checkpoint_path, cfg, device):
    """Load model from checkpoint"""
    print(f"\nLoading model from: {checkpoint_path}")

    if cfg.get("pass_device"):
        cfg["model_args"]["device"] = device

    model = HimNet(**cfg["model_args"])

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DataParallel wrapper if present
    if isinstance(checkpoint, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint.state_dict())
    else:
        # Check if checkpoint has 'module.' prefix (from DataParallel training)
        if all(key.startswith('module.') for key in checkpoint.keys()):
            # Create DataParallel wrapper
            model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def generate_predictions(model, runner, testset_loader):
    """Generate predictions on test set"""
    print("\nGenerating predictions...")

    with torch.no_grad():
        y_true, y_pred = runner.predict(model, testset_loader)

    # y_true and y_pred shape: [num_samples, seq_length, num_nodes]
    print(f"Original prediction shape: {y_pred.shape}")
    print(f"  - num_samples: {y_pred.shape[0]}")
    print(f"  - seq_length: {y_pred.shape[1]}")
    print(f"  - num_nodes: {y_pred.shape[2]}")

    # Transpose to required format: (seq_length, num_samples, num_nodes)
    y_pred_transposed = np.transpose(y_pred, (1, 0, 2))
    y_true_transposed = np.transpose(y_true, (1, 0, 2))

    print(f"\nTransposed prediction shape: {y_pred_transposed.shape}")
    print(f"  - seq_length: {y_pred_transposed.shape[0]}")
    print(f"  - num_samples: {y_pred_transposed.shape[1]}")
    print(f"  - num_nodes: {y_pred_transposed.shape[2]}")

    return y_true_transposed, y_pred_transposed


def save_npy_files(ground_truth, predictions, dataset, output_dir):
    """Save predictions and ground truth as NPY files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save ground truth
    gt_path = os.path.join(output_dir, 'ground_truth.npy')
    np.save(gt_path, ground_truth)
    print(f"\nSaved ground truth to: {gt_path}")
    print(f"  Shape: {ground_truth.shape}")
    print(f"  Dtype: {ground_truth.dtype}")
    print(f"  Value range: [{ground_truth.min():.4f}, {ground_truth.max():.4f}]")

    # Save HimNet predictions
    pred_path = os.path.join(output_dir, f'himnet_predictions.npy')
    np.save(pred_path, predictions)
    print(f"\nSaved HimNet predictions to: {pred_path}")
    print(f"  Shape: {predictions.shape}")
    print(f"  Dtype: {predictions.dtype}")
    print(f"  Value range: [{predictions.min():.4f}, {predictions.max():.4f}]")

    # Also save with dataset-specific name
    dataset_pred_path = os.path.join(output_dir, f'himnet_{dataset.lower()}_predictions.npy')
    np.save(dataset_pred_path, predictions)
    print(f"\nSaved dataset-specific predictions to: {dataset_pred_path}")

    return gt_path, pred_path


def main():
    # Parse arguments
    args = parse_args()

    # Set random seed
    seed_everything(args.seed)
    set_cpu_num(args.cpus)

    # Set device
    if torch.cuda.is_available() and args.device.startswith('cuda'):
        device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Dataset configuration
    dataset = args.dataset.upper()

    # Default model paths (best models)
    default_model_paths = {
        'METRLA': 'saved_models/HimNet-METRLA-2025-11-28-00-26-44_epoch24.pt',
        'PEMSBAY': 'saved_models/HimNet-PEMSBAY-2025-12-19-12-16-11_epoch13.pt'
    }

    print("="*80)
    print("HimNet NPY Prediction File Generator")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")

    # Load configuration
    config_path = os.path.join(project_root, "model", "HimNet.yaml")
    with open(config_path, "r") as f:
        all_cfg = yaml.safe_load(f)

    cfg = all_cfg[dataset]

    print(f"Configuration:")
    print(f"  Num Nodes: {cfg['model_args']['num_nodes']}")
    print(f"  Input Steps: {cfg['in_steps']}")
    print(f"  Output Steps: {cfg['out_steps']}")
    print("="*80 + "\n")

    # Load data
    print("Loading data...")
    trainset_loader, valset_loader, testset_loader, scaler = load_data(
        dataset, project_root, cfg
    )
    print("Data loaded successfully!\n")

    # Determine model path
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join(project_root, default_model_paths[dataset])

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("\nPlease specify the model path using --model_path argument.")
        print("Example:")
        print(f"  python generate_npy_predictions.py --dataset {dataset} --model_path /path/to/model.pt")
        sys.exit(1)

    print(f"Using model: {model_path}\n")

    # Load model
    model = load_model(model_path, cfg, device)

    # Set up runner
    runner = HimNetRunner(cfg, device=device, scaler=scaler, log=None)

    # Generate predictions
    ground_truth, predictions = generate_predictions(model, runner, testset_loader)

    # Save NPY files
    print("\n" + "="*80)
    print("Saving NPY files...")
    print("="*80)
    gt_path, pred_path = save_npy_files(ground_truth, predictions, dataset, args.output_dir)

    print("\n" + "="*80)
    print("NPY file generation completed successfully!")
    print("="*80)
    print(f"\nFiles saved to: {args.output_dir}")
    print(f"  - Ground truth: ground_truth.npy")
    print(f"  - HimNet predictions: himnet_predictions.npy")
    print(f"  - Dataset-specific: himnet_{dataset.lower()}_predictions.npy")
    print("\nFormat verification:")
    print(f"  Shape: (seq_length={predictions.shape[0]}, num_samples={predictions.shape[1]}, num_nodes={predictions.shape[2]})")
    print(f"  Dtype: {predictions.dtype}")
    print("="*80)


if __name__ == '__main__':
    main()
