#!/usr/bin/env python3
"""
Generate STGCN prediction results in NPY format
Output format: (seq_length, num_samples, num_nodes)
"""

import sys
sys.path.append('../')

import os
import argparse
import configparser
import torch
import numpy as np
from lib.utils import *
from lib.data_loader import *
from lib.generate_adj_mx import *
from STGCN_Utils import *
from model.STGCN.stgcn import STGCN as Network


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate STGCN predictions in NPY format')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['METRLA', 'PEMSBAY'],
                        help='Dataset to use: METRLA or PEMSBAY')

    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model checkpoint. If not specified, will use default path.')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use: cuda:0, cuda:1, or cpu')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for testing')

    parser.add_argument('--output_dir', type=str, default='./predictions_npy',
                        help='Directory to save NPY files')

    args = parser.parse_args()
    return args


def setup_dataset_config(dataset_name):
    """
    Load dataset configuration from config file

    Returns:
        dict: Configuration dictionary
    """
    base_configs = {
        'METRLA': {
            'dataset_dir': '../data/METR-LA/processed/',
            'graph_pkl': '../data/METR-LA/processed/adj_mx.pkl',
            'config_file': './METRLA_STGCN.conf',
            'default_model_path': '../log/STGCN/METRLA/*/METRLA_STGCN_best_model.pth'
        },
        'PEMSBAY': {
            'dataset_dir': '../data/PEMS-BAY/processed/',
            'graph_pkl': '../data/PEMS-BAY/processed/adj_mx_bay.pkl',
            'config_file': './PEMSBAY_STGCN.conf',
            'default_model_path': '../log/STGCN/PEMSBAY/*/PEMSBAY_STGCN_best_model.pth'
        }
    }

    if dataset_name not in base_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    base_config = base_configs[dataset_name]

    # Read config file
    config_file = base_config['config_file']
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config = configparser.ConfigParser()
    config.read(config_file)

    # Parse configuration
    full_config = base_config.copy()
    full_config['num_nodes'] = int(config['data']['num_nodes'])
    full_config['window'] = int(config['data']['window'])
    full_config['horizon'] = int(config['data']['horizon'])
    full_config['input_dim'] = eval(config['model']['input_dim'])
    full_config['KS'] = eval(config['model']['KS'])
    full_config['KT'] = eval(config['model']['KT'])
    full_config['channels'] = eval(config['model']['channels'])
    full_config['dropout'] = eval(config['model']['dropout'])

    return full_config


def find_latest_model(pattern):
    """
    Find the latest model file matching the pattern

    Args:
        pattern: Model path pattern (can include wildcards)

    Returns:
        str: Latest model file path
    """
    import glob

    if '*' in pattern:
        matching_files = glob.glob(pattern)
        if not matching_files:
            return None
        # Sort by modification time, return the latest
        matching_files.sort(key=os.path.getmtime, reverse=True)
        return matching_files[0]
    else:
        return pattern if os.path.exists(pattern) else None


class SimpleArgs:
    """Simple argument class"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def load_data(args):
    """Load dataset and adjacency matrix"""
    data_loader = load_dataset(args.dataset_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler = data_loader['scaler']

    # Load adjacency matrix
    _, _, adj_mx = load_pickle(args.graph_pkl)
    adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    return adj_mx, data_loader, scaler


def generate_model(args, adj_mx):
    """Generate STGCN model"""
    L = scaled_laplacian(adj_mx)
    Lk = cheb_poly(L, Ks=args.KS)
    Lk = torch.Tensor(Lk.astype(np.float32)).to(args.device)

    model = Network(
        ks=args.KS,
        kt=args.KT,
        bs=args.channels,
        T=args.window,
        n=args.num_nodes,
        Lk=Lk,
        p=args.dropout,
        horizon=args.horizon
    )
    model = model.to(args.device)

    return model


def generate_predictions(args, model, data_loader, scaler, model_path):
    """
    Generate predictions and ground truth in the required NPY format

    Returns:
        predictions: numpy array of shape (seq_length, num_samples, num_nodes)
        ground_truth: numpy array of shape (seq_length, num_samples, num_nodes)
    """
    # Load model
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    model.eval()
    print(f"Loaded model from: {model_path}")

    # Prepare test data
    test_loader = data_loader['test_loader']
    realy = torch.Tensor(data_loader['y_test']).to(args.device)
    realy = realy[:, :, :, 0:1]  # (B, T, N, 1)

    # Generate predictions
    outputs = []
    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader.get_iterator()):
            testx = torch.Tensor(x).to(args.device)
            output = model(testx[:, :, :, :args.input_dim])
            outputs.append(output)

    # Concatenate all batches
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    # Collect predictions and ground truth for each time step
    predictions_list = []
    ground_truth_list = []

    print("\nGenerating predictions for each time step...")
    for i in range(args.horizon):
        # Inverse transform predictions to get real values
        pred = scaler.inverse_transform(yhat[:, i, :, :])  # (B, N, 1)
        real = realy[:, i, :, :]  # (B, N, 1)

        # Remove the last dimension and convert to numpy
        pred_np = pred.squeeze(-1).cpu().numpy()  # (B, N)
        real_np = real.squeeze(-1).cpu().numpy()  # (B, N)

        predictions_list.append(pred_np)
        ground_truth_list.append(real_np)

        if (i + 1) % 12 == 0 or i == args.horizon - 1:
            print(f"  Processed time step {i+1}/{args.horizon}")

    # Stack to get shape (T, B, N)
    predictions = np.stack(predictions_list, axis=0)  # (T, B, N)
    ground_truth = np.stack(ground_truth_list, axis=0)  # (T, B, N)

    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Data type: {predictions.dtype}")

    # Verify the format
    seq_length, num_samples, num_nodes = predictions.shape
    print(f"\nFormat verification:")
    print(f"  seq_length (horizon): {seq_length}")
    print(f"  num_samples: {num_samples}")
    print(f"  num_nodes: {num_nodes}")

    return predictions, ground_truth


def save_npy_files(predictions, ground_truth, dataset_name, output_dir):
    """
    Save predictions and ground truth as NPY files

    Args:
        predictions: numpy array of shape (seq_length, num_samples, num_nodes)
        ground_truth: numpy array of shape (seq_length, num_samples, num_nodes)
        dataset_name: name of the dataset (METRLA or PEMSBAY)
        output_dir: directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    pred_filename = f'stgcn_{dataset_name.lower()}_predictions.npy'
    pred_path = os.path.join(output_dir, pred_filename)
    np.save(pred_path, predictions)
    print(f"\nSaved predictions to: {pred_path}")

    # Save ground truth
    gt_filename = f'{dataset_name.lower()}_ground_truth.npy'
    gt_path = os.path.join(output_dir, gt_filename)
    np.save(gt_path, ground_truth)
    print(f"Saved ground truth to: {gt_path}")

    # Print file information
    print(f"\nFile information:")
    print(f"  Predictions file: {pred_filename}")
    print(f"    - Shape: {predictions.shape}")
    print(f"    - Dtype: {predictions.dtype}")
    print(f"    - Size: {os.path.getsize(pred_path) / (1024*1024):.2f} MB")
    print(f"    - Value range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    print(f"\n  Ground truth file: {gt_filename}")
    print(f"    - Shape: {ground_truth.shape}")
    print(f"    - Dtype: {ground_truth.dtype}")
    print(f"    - Size: {os.path.getsize(gt_path) / (1024*1024):.2f} MB")
    print(f"    - Value range: [{ground_truth.min():.2f}, {ground_truth.max():.2f}]")


def compute_metrics(predictions, ground_truth):
    """Compute and display basic metrics"""
    mae = np.mean(np.abs(predictions - ground_truth))
    rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
    mape = np.mean(np.abs((predictions - ground_truth) / (ground_truth + 1e-5))) * 100

    print(f"\nOverall Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.4f}%")


def main():
    # Parse arguments
    cmd_args = parse_args()

    # Setup dataset configuration
    dataset_config = setup_dataset_config(cmd_args.dataset)

    # Create unified args object
    args = SimpleArgs(
        dataset=cmd_args.dataset,
        device=cmd_args.device,
        batch_size=cmd_args.batch_size,
        dataset_dir=dataset_config['dataset_dir'],
        graph_pkl=dataset_config['graph_pkl'],
        num_nodes=dataset_config['num_nodes'],
        window=dataset_config['window'],
        horizon=dataset_config['horizon'],
        input_dim=dataset_config['input_dim'],
        KS=dataset_config['KS'],
        KT=dataset_config['KT'],
        channels=dataset_config['channels'],
        dropout=dataset_config['dropout']
    )

    # Setup device
    if torch.cuda.is_available() and args.device.startswith('cuda'):
        device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        torch.cuda.set_device(device_id)
    else:
        args.device = 'cpu'

    print("="*80)
    print("STGCN Prediction NPY Generator")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Num Nodes: {args.num_nodes}")
    print(f"Window: {args.window}")
    print(f"Horizon: {args.horizon}")
    print(f"Output Dir: {cmd_args.output_dir}")
    print("="*80 + "\n")

    # Load data
    print("Loading data...")
    adj_mx, data_loader, scaler = load_data(args)
    print("Data loaded successfully!\n")

    # Generate model
    print("Generating model...")
    model = generate_model(args, adj_mx)
    print("Model generated successfully!\n")

    # Determine model path
    if cmd_args.model_path is not None:
        model_path = cmd_args.model_path
    else:
        # Use default path pattern to find latest model
        model_path = find_latest_model(dataset_config['default_model_path'])

    if model_path is None or not os.path.exists(model_path):
        print(f"Error: Model file not found!")
        if cmd_args.model_path is not None:
            print(f"  Specified path: {cmd_args.model_path}")
        else:
            print(f"  Searched pattern: {dataset_config['default_model_path']}")
        print("\nPlease specify the model path using --model_path argument.")
        print("Example:")
        print(f"  python generate_predictions_npy.py --dataset {args.dataset} --model_path /path/to/model.pth")
        sys.exit(1)

    print(f"Using model: {model_path}\n")

    # Generate predictions
    print("Generating predictions...")
    predictions, ground_truth = generate_predictions(args, model, data_loader, scaler, model_path)

    # Compute metrics
    compute_metrics(predictions, ground_truth)

    # Save NPY files
    print("\nSaving NPY files...")
    save_npy_files(predictions, ground_truth, args.dataset, cmd_args.output_dir)

    print("\n" + "="*80)
    print("NPY generation completed successfully!")
    print(f"Files saved to: {cmd_args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
