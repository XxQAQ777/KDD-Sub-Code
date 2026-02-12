#!/usr/bin/env python3
"""
Enhanced test script for HimNet with visualization capabilities.
Based on test_checkpoint.py with plotting features inspired by test_new.py
"""

import argparse
import numpy as np
import os
import torch
import datetime
import time
import yaml
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.utils import (
    print_log,
    seed_everything,
    set_cpu_num,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from model.HimNet import HimNet
from model.HimNetRunner import HimNetRunner


def plot_node_predictions(node_idx, sample_idx, y_true, y_pred, seq_length, fig_dir="figure"):
    """
    Plot predictions vs ground truth for a specific node and sample.

    Features:
    - No grid, clean design
    - X-axis shows ticks every 12 steps
    - Optimized colors (deep cyan for truth, brick red for prediction)

    Args:
        node_idx: Index of the node to plot
        sample_idx: Index of the sample to plot
        y_true: Ground truth values [num_samples, seq_length, num_nodes]
        y_pred: Predicted values [num_samples, seq_length, num_nodes]
        seq_length: Prediction horizon length
        fig_dir: Directory to save figures

    Returns:
        Path to saved figure or None if invalid indices
    """
    # Boundary check
    if node_idx < 0 or node_idx >= y_true.shape[2]:
        return None
    if sample_idx < 0 or sample_idx >= y_true.shape[0]:
        return None

    # Extract data for the specific node and sample
    real_values = y_true[sample_idx, :, node_idx]  # [seq_length]
    pred_values = y_pred[sample_idx, :, node_idx]  # [seq_length]

    # Create directory if not exists
    os.makedirs(fig_dir, exist_ok=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    horizons = np.arange(1, seq_length + 1)

    # Plot ground truth: deep cyan (#005f73), solid line
    ax.plot(horizons, real_values, label='Ground Truth',
            color='#005f73', linestyle='-', linewidth=2, alpha=0.9)

    # Plot prediction: brick red (#ae2012), dashed line
    ax.plot(horizons, pred_values, label='Prediction',
            color='#ae2012', linestyle='--', linewidth=2, alpha=0.9)

    # Set title and labels
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

    # Legend: no frame, upper right
    ax.legend(frameon=False, loc='upper right', fontsize=11)

    # Save figure
    fig_path = os.path.join(fig_dir, f'node_{node_idx}_sample_{sample_idx}.png')
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    return fig_path


def plot_all_steps_comparison(y_true, y_pred, dataset_name, fig_dir="figure"):
    """
    Plot average MAE across all time steps.

    Args:
        y_true: Ground truth [num_samples, seq_length, num_nodes]
        y_pred: Predictions [num_samples, seq_length, num_nodes]
        dataset_name: Name of the dataset
        fig_dir: Directory to save figure
    """
    seq_length = y_true.shape[1]
    mae_per_step = []

    for i in range(seq_length):
        mae = np.mean(np.abs(y_true[:, i, :] - y_pred[:, i, :]))
        mae_per_step.append(mae)

    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    horizons = np.arange(1, seq_length + 1)

    ax.plot(horizons, mae_per_step, color='#005f73', linewidth=2, marker='o', markersize=3)
    ax.set_title(f'MAE vs Prediction Horizon - {dataset_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Prediction Horizon (Steps)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_ylim(bottom=0)

    # X-axis ticks every 12 steps
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12))

    # Clean design
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    fig_path = os.path.join(fig_dir, f'{dataset_name}_mae_per_horizon.png')
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    return fig_path


def test_checkpoint_with_plot(
    checkpoint_path,
    dataset="METRLA",
    gpu_num=0,
    seed=0,
    cpus=1,
    plot_all=False,
    plot_node=-1,
    plot_sample_idx=0,
    fig_dir="figure_predictions"
):
    """
    Test a specific checkpoint on the test set with visualization options.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset: Dataset name (METRLA or PEMSBAY)
        gpu_num: GPU device number
        seed: Random seed
        cpus: Number of CPU cores
        plot_all: If True, plot all nodes for the specified sample
        plot_node: If >= 0, plot only this specific node
        plot_sample_idx: Sample index to use for plotting
        fig_dir: Directory to save figures
    """
    if not seed:
        seed = np.random.randint(1, 10000)

    seed_everything(seed)
    set_cpu_num(cpus)

    # Set CUDA visible devices
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
        print(f"Using GPU: {gpu_num}")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = dataset.upper()

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", dataset)

    model_name = "HimNet"

    config_path = os.path.join(project_root, "model", f"{model_name}.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #
    print(f"\n{'='*80}")
    print(f"HimNet Testing with Visualization")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset}")
    print(f"{'='*80}\n")

    if cfg.get("pass_device"):
        cfg["model_args"]["device"] = DEVICE

    model = HimNet(**cfg["model_args"])

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

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

    model.to(DEVICE)
    model.eval()

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(project_root, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f"test-plot-{model_name}-{dataset}-{now}.log")
    log = open(log_file, "w")

    # ------------------------------- load dataset ------------------------------- #

    print_log(f"Testing checkpoint: {checkpoint_path}", log=log)
    print_log(f"Dataset: {dataset}", log=log)

    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        y_tod=cfg.get("y_time_of_day"),
        y_dow=cfg.get("y_day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)

    # ----------------------------- set model runner ----------------------------- #

    runner = HimNetRunner(cfg, device=DEVICE, scaler=SCALER, log=log)

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(f"Seed = {seed}", log=log)
    print_log(f"Dataset: {dataset}", log=log)
    print_log(f"Input steps: {cfg.get('in_steps', 'N/A')}", log=log)
    print_log(f"Output steps: {cfg.get('out_steps', 'N/A')}", log=log)
    print_log(f"Device: {DEVICE}", log=log)
    print_log(f"Using {torch.cuda.device_count()} GPU(s)" if torch.cuda.is_available() else "Using CPU", log=log)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_log(f"Total parameters: {total_params:,}", log=log)
    print_log(f"Trainable parameters: {trainable_params:,}", log=log)
    print_log(log=log)

    # --------------------------- test model --------------------------- #

    print_log("--------- Test Results ---------", log=log)

    start = time.time()
    y_true, y_pred = runner.predict(model, testset_loader)
    end = time.time()

    # y_true and y_pred shape: [num_samples, seq_length, num_nodes]
    print_log(f"Prediction shape: {y_pred.shape}", log=log)

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)

    # --------------------------- Visualization --------------------------- #

    num_nodes = cfg['model_args']['num_nodes']
    seq_length = cfg.get('out_steps', y_pred.shape[1])

    # Create figure directory
    fig_output_dir = os.path.join(project_root, fig_dir)
    os.makedirs(fig_output_dir, exist_ok=True)

    print_log("\n--------- Visualization ---------", log=log)
    print_log(f"Figure output directory: {fig_output_dir}", log=log)

    # Plot MAE per horizon
    print_log("Plotting MAE per prediction horizon...", log=log)
    mae_plot_path = plot_all_steps_comparison(y_true, y_pred, dataset, fig_output_dir)
    print_log(f"Saved: {mae_plot_path}", log=log)
    print(f"Saved MAE plot: {mae_plot_path}")

    # Plot node predictions
    if plot_all:
        print_log(f"\nPlotting all {num_nodes} nodes (Sample ID: {plot_sample_idx})", log=log)
        print(f"\nPlotting all {num_nodes} nodes (Sample ID: {plot_sample_idx})")
        print("This may take some time, please wait...")

        count = 0
        for node_idx in range(num_nodes):
            fig_path = plot_node_predictions(
                node_idx, plot_sample_idx, y_true, y_pred, seq_length, fig_output_dir
            )
            if fig_path:
                count += 1
                if count % 20 == 0:
                    progress_msg = f"  Plotted {count}/{num_nodes} nodes..."
                    print_log(progress_msg, log=log)
                    print(progress_msg)

        print_log(f"Completed plotting {count} nodes", log=log)
        print(f"Completed plotting {count} nodes. Check {fig_output_dir} folder.")

    elif plot_node >= 0:
        print_log(f"\nPlotting single node {plot_node} (Sample ID: {plot_sample_idx})", log=log)
        print(f"\nPlotting single node {plot_node} (Sample ID: {plot_sample_idx})")

        fig_path = plot_node_predictions(
            plot_node, plot_sample_idx, y_true, y_pred, seq_length, fig_output_dir
        )

        if fig_path:
            print_log(f"Saved: {fig_path}", log=log)
            print(f"Saved: {fig_path}")
        else:
            print_log(f"WARNING: Invalid node index {plot_node} or sample index {plot_sample_idx}", log=log)
            print(f"WARNING: Invalid node index {plot_node} or sample index {plot_sample_idx}")

    # Also print to console
    print("\n" + "="*80)
    print(f"Test Results for {checkpoint_path}")
    print("="*80)
    print(out_str)
    print(f"Inference time: {end - start:.2f} s")
    print(f"Detailed log saved to: {log_file}")
    print("="*80 + "\n")

    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test HimNet checkpoint with visualization options"
    )
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("-d", "--dataset", type=str, default="METRLA",
                       help="Dataset name (METRLA or PEMSBAY)")
    parser.add_argument("-g", "--gpu_num", type=int, default=0,
                       help="GPU number to use")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--cpus", type=int, default=1,
                       help="Number of CPUs to use")

    # Visualization options
    parser.add_argument("--plot_node", type=int, default=-1,
                       help="Plot specific node index (ignored if --all is set)")
    parser.add_argument("--plot_sample_idx", type=int, default=0,
                       help="Sample index to use for plotting")
    parser.add_argument("--all", action="store_true",
                       help="Plot all nodes for the specified sample")
    parser.add_argument("--fig_dir", type=str, default="figure_predictions",
                       help="Directory to save figures")

    args = parser.parse_args()

    test_checkpoint_with_plot(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        gpu_num=args.gpu_num,
        seed=args.seed,
        cpus=args.cpus,
        plot_all=args.all,
        plot_node=args.plot_node,
        plot_sample_idx=args.plot_sample_idx,
        fig_dir=args.fig_dir
    )
