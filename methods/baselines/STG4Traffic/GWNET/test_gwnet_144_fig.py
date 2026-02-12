#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GWNET 144-timestep Testing and Visualization Script

Usage:
    python test_gwnet_144_fig.py \
        --ckpt ../log/GWNET/METRLA_144/YOUR_MODEL.pth \
        --data ../data/METR-LA-144-3feat \
        --adjdata ../data/sensor_graph/adj_mx.pkl \
        --all \
        --fig_dir figure_gwnet_144

Author: Generated for GWNET 144-timestep prediction
Date: 2026-01-04
"""

import sys
sys.path.append('../')

import argparse
import os
import time
import traceback
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from lib.utils import load_pickle, asym_adj, metric
from lib.data_loader import load_dataset, StandardScaler
from model.GWNET.gwnet import gwnet


def log(msg):
    """Print log message with flush"""
    print(msg, flush=True)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='GWNET 144-timestep Testing with Visualization')

    # Model and data paths
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='../data/METR-LA-144-3feat',
                        help='Path to dataset directory')
    parser.add_argument('--adjdata', type=str, default='../data/sensor_graph/adj_mx.pkl',
                        help='Path to adjacency matrix pickle file')
    parser.add_argument('--adjtype', type=str, default='doubletransition',
                        help='Adjacency matrix type')

    # Model parameters
    parser.add_argument('--num_nodes', type=int, default=207,
                        help='Number of nodes')
    parser.add_argument('--seq_length', type=int, default=144,
                        help='Sequence length (input and output timesteps)')
    parser.add_argument('--in_dim', type=int, default=2,
                        help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension (residual_channels)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--blocks', type=int, default=6,
                        help='Number of blocks')
    parser.add_argument('--layers', type=int, default=3,
                        help='Number of layers per block')

    # GCN parameters
    parser.add_argument('--gcn_bool', action='store_true', default=True,
                        help='Enable graph convolution')
    parser.add_argument('--addaptadj', action='store_true', default=True,
                        help='Enable adaptive adjacency matrix')

    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu)')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Print frequency during testing')

    # Visualization parameters
    parser.add_argument('--plot_node', type=int, default=20,
                        help='Node index to plot (used when --all is not set)')
    parser.add_argument('--plot_sample_idx', type=int, default=25,
                        help='Sample index to plot')
    parser.add_argument('--all', action='store_true',
                        help='Plot all nodes (overrides --plot_node)')
    parser.add_argument('--fig_dir', type=str, default='figure_gwnet_144',
                        help='Directory to save figures')

    return parser.parse_args()


def set_device(device_str):
    """Set and return device"""
    if torch.cuda.is_available() and device_str.startswith('cuda'):
        return torch.device(device_str)
    log(f"WARNING: CUDA not available or device {device_str} invalid, using CPU")
    return torch.device('cpu')


def load_adjacency_matrix(adj_path, adj_type='doubletransition'):
    """Load and process adjacency matrix"""
    log(f"Loading adjacency matrix from: {adj_path}")
    _, _, adj_mx = load_pickle(adj_path)

    if adj_type == 'doubletransition':
        supports = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    else:
        raise ValueError(f"Unsupported adjacency type: {adj_type}")

    return supports


def build_model(args, device, supports):
    """Build GWNET model"""
    log("Building GWNET model...")

    # Convert supports to tensors
    supports_tensor = [torch.tensor(adj, dtype=torch.float32).to(device) for adj in supports]

    model = gwnet(
        device=device,
        num_nodes=args.num_nodes,
        dropout=args.dropout,
        supports=supports_tensor,
        gcn_bool=args.gcn_bool,
        addaptadj=args.addaptadj,
        aptinit=None,  # Can use supports_tensor[0] for initialization
        in_dim=args.in_dim,
        out_dim=args.seq_length,
        residual_channels=args.hidden_dim,
        dilation_channels=args.hidden_dim,
        skip_channels=8 * args.hidden_dim,
        end_channels=16 * args.hidden_dim,
        blocks=args.blocks,
        layers=args.layers
    )

    model = model.to(device)

    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total parameters: {total_params:,}")
    log(f"Trainable parameters: {trainable_params:,}")

    return model


def load_model_checkpoint(model, ckpt_path, device):
    """Load model checkpoint"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    log(f"Loading checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)

    try:
        model.load_state_dict(state_dict, strict=True)
        log("Model loaded successfully (strict=True)")
    except Exception as e:
        log(f"WARNING: Strict loading failed: {e}")
        log("Trying to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        log("Model loaded with strict=False")

    return model


def plot_node_predictions(node_idx, sample_idx, realy_eval, yhatt, scaler, seq_length, fig_dir="figure"):
    """
    Plot predictions vs ground truth for a specific node and sample.

    Args:
        node_idx: Node index to plot
        sample_idx: Sample index to plot
        realy_eval: Ground truth values (already in original scale) [num_samples, num_nodes, seq_length]
        yhatt: Predicted values (normalized scale) [num_samples, num_nodes, seq_length]
        scaler: StandardScaler for inverse transformation
        seq_length: Sequence length
        fig_dir: Directory to save figures
    """
    # Boundary checks
    if node_idx < 0 or node_idx >= realy_eval.shape[1]:
        return
    if sample_idx < 0 or sample_idx >= realy_eval.shape[0]:
        return

    # Prepare data
    real_values_unscaled = []
    pred_values_unscaled = []

    for h in range(seq_length):
        # Ground truth (already in original scale)
        real_val = realy_eval[sample_idx, node_idx, h].item()
        real_values_unscaled.append(real_val)

        # Prediction (normalized -> inverse transform)
        pred_tensor_step = yhatt[sample_idx, :, h]
        pred_tensor_batch = pred_tensor_step.unsqueeze(0)  # [1, num_nodes]
        pred_unscaled_batch = scaler.inverse_transform(pred_tensor_batch)

        if torch.is_tensor(pred_unscaled_batch):
            pred_val = pred_unscaled_batch[0, node_idx].item()
        else:
            pred_val = pred_unscaled_batch[0, node_idx]

        pred_values_unscaled.append(pred_val)

    # Convert to numpy
    real_values_unscaled = np.array(real_values_unscaled)
    pred_values_unscaled = np.array(pred_values_unscaled)

    # Create directory
    os.makedirs(fig_dir, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    horizons = np.arange(1, seq_length + 1)

    # Ground truth: deep cyan
    ax.plot(horizons, real_values_unscaled, label='Ground Truth',
            color='#005f73', linestyle='-', linewidth=2, alpha=0.9)

    # Prediction: brick red
    ax.plot(horizons, pred_values_unscaled, label='Prediction',
            color='#ae2012', linestyle='--', linewidth=2, alpha=0.9)

    # Formatting
    ax.set_title(f'Node {node_idx} - Sample {sample_idx}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Traffic Speed', fontsize=12)
    ax.set_ylim(bottom=0)

    # X-axis ticks every 12 steps
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    # No grid
    ax.grid(False)

    # Legend
    ax.legend(frameon=False, loc='upper right', fontsize=11)

    # Save figure
    fig_path = os.path.join(fig_dir, f'node_{node_idx}_sample_{sample_idx}.png')
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    return fig_path


def test_and_visualize(args):
    """Main testing and visualization function"""
    t0 = time.time()
    log("="*80)
    log("GWNET 144-Timestep Testing and Visualization")
    log("="*80)
    log(f"Checkpoint: {args.ckpt}")
    log(f"Data: {args.data}")
    log(f"Device: {args.device}")

    # Set device
    device = set_device(args.device)

    # Load adjacency matrix
    supports = load_adjacency_matrix(args.adjdata, args.adjtype)
    log(f"Adjacency matrix loaded. Number of support matrices: {len(supports)}")

    # Load dataset
    log(f"Loading dataset from: {args.data}")
    dataloader = load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    y_test = dataloader.get('y_test', None)

    if y_test is None:
        raise ValueError("Test labels (y_test) not found in dataloader")

    log(f"Dataset loaded. Test samples: {y_test.shape[0]}")
    log(f"Data shape - X: {dataloader['x_test'].shape}, Y: {y_test.shape}")

    # Build model
    model = build_model(args, device, supports)

    # Load checkpoint
    model = load_model_checkpoint(model, args.ckpt, device)
    model.eval()

    # Prepare ground truth
    # y_test shape: (num_samples, seq_length, num_nodes, feature_dim)
    # We only need the first feature dimension (traffic speed)
    realy = torch.tensor(y_test[:, :, :, 0], dtype=torch.float32, device=device)  # [B, T, N]
    realy = realy.permute(0, 2, 1)  # [B, N, T]

    log(f"Ground truth shape: {realy.shape}")

    # Get test iterator
    test_iter = dataloader['test_loader'].get_iterator()

    # Inference
    log("Starting inference...")
    outputs = []
    batch_count = 0

    inference_start = time.time()
    with torch.no_grad():
        for x, y in test_iter:
            testx = torch.tensor(x, dtype=torch.float32, device=device)
            # testx shape: [B, T, N, F] -> need to transpose for model
            testx = testx.permute(0, 3, 2, 1)  # [B, F, N, T]

            # Forward pass
            preds = model(testx)  # [B, out_dim, N, 1]

            # Reshape predictions
            preds = preds.squeeze(-1)  # [B, out_dim, N]
            preds = preds.permute(0, 2, 1)  # [B, N, out_dim]

            outputs.append(preds)
            batch_count += 1

            if batch_count % args.print_every == 0:
                log(f"Processed batch {batch_count}")

    inference_time = time.time() - inference_start
    log(f"Inference completed in {inference_time:.2f}s")

    # Concatenate predictions
    yhatt = torch.cat(outputs, dim=0)  # [total_samples, N, T]

    # Align lengths
    min_samples = min(yhatt.size(0), realy.size(0))
    realy_eval = realy[:min_samples]
    yhatt = yhatt[:min_samples]

    log(f"Prediction shape: {yhatt.shape}")
    log(f"Evaluation samples: {min_samples}")

    # Calculate metrics for each horizon
    log("\n" + "="*80)
    log("Evaluation Metrics")
    log("="*80)

    amae, amape, armse = [], [], []

    for h in range(args.seq_length):
        # Inverse transform predictions
        pred = scaler.inverse_transform(yhatt[:, :, h])
        real = realy_eval[:, :, h]

        # Calculate metrics
        mae, mape, rmse = metric(pred, real)

        # Log every 12 horizons or first/last
        if h % 12 == 0 or h == args.seq_length - 1:
            log(f"Horizon {h+1:3d}: MAE={mae:.4f}, MAPE={mape:.4f}, RMSE={rmse:.4f}")

        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)

    log("="*80)
    log(f"Average over {args.seq_length} horizons:")
    log(f"  MAE:  {np.mean(amae):.4f}")
    log(f"  MAPE: {np.mean(amape):.4f}")
    log(f"  RMSE: {np.mean(armse):.4f}")
    log("="*80)

    # Visualization
    log("\n" + "="*80)
    log("Visualization")
    log("="*80)

    if args.all:
        log(f"Plotting all {args.num_nodes} nodes (Sample ID: {args.plot_sample_idx})")
        log(f"Output directory: {args.fig_dir}")
        log("This may take a while...")

        count = 0
        for n_idx in range(args.num_nodes):
            plot_node_predictions(n_idx, args.plot_sample_idx, realy_eval, yhatt,
                                 scaler, args.seq_length, fig_dir=args.fig_dir)
            count += 1
            if count % 20 == 0:
                log(f"  Plotted {count}/{args.num_nodes} nodes...")

        log(f"All plots saved to: {args.fig_dir}/")
    else:
        log(f"Plotting node {args.plot_node} (Sample ID: {args.plot_sample_idx})")
        fig_path = plot_node_predictions(args.plot_node, args.plot_sample_idx,
                                         realy_eval, yhatt, scaler, args.seq_length,
                                         fig_dir=args.fig_dir)
        log(f"Plot saved to: {fig_path}")

    log("="*80)
    log(f"Total execution time: {(time.time() - t0):.2f}s")
    log("="*80)


def main():
    """Main entry point"""
    try:
        args = parse_args()
        test_and_visualize(args)
    except Exception as e:
        log("\n" + "="*80)
        log("ERROR: An exception occurred during execution")
        log("="*80)
        traceback.print_exc()
        log("="*80)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
