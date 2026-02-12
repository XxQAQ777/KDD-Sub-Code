"""
使用说明：
一键测试PEMS-BAY数据集（默认节点0,1,2,3）：
python test_all_fig_stgcn_style.py \
    --dataset pems \
    --model default \
    --ckpt gara_1222/PEMS_nhid=16_exp1_best_2.26.pth

一键测试METR-LA数据集：
python test_all_fig_stgcn_style.py \
    --dataset metr \
    --model composite_loss \
    --ckpt gara_1229old/metr_patch=4_epoch_10_3.99.pth

指定特定节点和样本：
python test_all_fig_stgcn_style.py \
    --dataset pems \
    --model default \
    --ckpt gara_1222/PEMS_nhid=16_exp1_best_2.26.pth \
    --plot_nodes "10,50,100,200" \
    --plot_sample_idx 25
"""
import argparse
import os
import sys
import time
import traceback
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime
from scipy.stats import wasserstein_distance
import util
import importlib


def log(msg):
    print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()

    # 新增：数据集快捷选项
    parser.add_argument('--dataset', type=str, choices=['pems', 'metr'], default=None,
                        help='快捷选择数据集 (pems或metr)，会自动配置data/adjdata/num_nodes')

    # 模型相关
    parser.add_argument('--model', type=str, default='default', help='model name (e.g., default, composite_loss)')
    parser.add_argument('--ckpt', type=str, required=True, help='模型检查点路径')

    # 数据相关（如果使用--dataset，这些会被自动设置）
    parser.add_argument('--data', type=str, default=None, help='数据路径')
    parser.add_argument('--adjdata', type=str, default=None, help='邻接矩阵路径')
    parser.add_argument('--num_nodes', type=int, default=None, help='节点数')

    # 其他参数
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='邻接类型')
    parser.add_argument('--seq_length', type=int, default=144, help='预测步长')
    parser.add_argument('--in_dim', type=int, default=3, help='输入维度')
    parser.add_argument('--nhid', type=int, default=16, help='隐藏通道')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--gcn_bool', action='store_true', help='是否使用图卷积')
    parser.add_argument('--aptonly', action='store_true', help='仅使用自适应邻接')
    parser.add_argument('--addaptadj', action='store_true', help='启用自适应邻接')
    parser.add_argument('--randomadj', action='store_true', help='自适应邻接随机初始化')
    parser.add_argument('--batch_size', type=int, default=16, help='测试批大小')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--print_every', type=int, default=100, help='打印频率')
    parser.add_argument('--fig_dir', type=str, default=None, help='图片保存文件夹')
    parser.add_argument('--plot_nodes', type=str, default=None,
                        help='指定要绘制的节点列表，逗号分隔，例如 "0,5,10,15"。默认选择前4个节点')
    parser.add_argument('--plot_sample_idx', type=int, default=0, help='用于时间序列绘图的样本索引')

    args = parser.parse_args()

    # 根据dataset参数自动配置
    if args.dataset == 'pems':
        if args.data is None:
            args.data = 'data/PEMS-BAY-144-3feat-row'
        if args.adjdata is None:
            args.adjdata = 'data/sensor_graph/adj_mx_bay.pkl'
        if args.num_nodes is None:
            args.num_nodes = 325
        if args.fig_dir is None:
            args.fig_dir = 'test_results_pems_stgcn'
    elif args.dataset == 'metr':
        if args.data is None:
            args.data = 'data/METR-LA-144-3feat'
        if args.adjdata is None:
            args.adjdata = 'data/sensor_graph/adj_mx.pkl'
        if args.num_nodes is None:
            args.num_nodes = 207
        if args.fig_dir is None:
            args.fig_dir = 'test_results_metr_stgcn'
    else:
        # 如果没有指定dataset，检查必需参数
        if args.data is None:
            args.data = 'data/METR-LA-144-3feat'
        if args.adjdata is None:
            args.adjdata = 'data/sensor_graph/adj_mx.pkl'
        if args.num_nodes is None:
            args.num_nodes = 207
        if args.fig_dir is None:
            args.fig_dir = 'test_results_stgcn_style'

    return args


def set_device(device_str):
    if torch.cuda.is_available() and device_str.startswith('cuda'):
        return torch.device(device_str)
    return torch.device('cpu')


def compute_crps(pred, real):
    """
    计算CRPS (Continuous Ranked Probability Score)
    对于确定性预测，CRPS简化为MAE
    pred, real: torch tensors of shape (num_samples, num_nodes)
    """
    # 对于确定性预测，CRPS = MAE
    crps = torch.mean(torch.abs(pred - real)).item()
    return crps


def compute_wasserstein(pred, real):
    """
    计算Wasserstein Distance (Earth Mover's Distance)
    pred, real: torch tensors of shape (num_samples, num_nodes)
    """
    pred_np = pred.cpu().numpy().flatten()
    real_np = real.cpu().numpy().flatten()

    # 使用scipy的wasserstein_distance
    # 这计算的是1-Wasserstein距离
    wd = wasserstein_distance(pred_np, real_np)
    return wd


def build_model(args, device, supports, model_module):
    adjinit = None if args.randomadj else (supports[0] if supports is not None else None)
    model = model_module.gwnet(
        device, args.num_nodes, args.dropout,
        supports=None if args.aptonly else supports,
        gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit,
        in_dim=args.in_dim, out_dim=args.seq_length,
        residual_channels=args.nhid, dilation_channels=args.nhid,
        skip_channels=args.nhid * 8, end_channels=args.nhid * 16
    )
    model.to(device)
    return model


def test_model(args, model, dataloader, scaler, device):
    """测试模型并返回预测结果和真实值（使用test_all_fig.py的逻辑）"""
    log("Starting model inference...")
    model.eval()

    # 准备测试数据（与test_all_fig.py一致）
    test_loader = dataloader['test_loader']
    y_test = dataloader.get('y_test', None)

    if y_test is None:
        log("ERROR: dataloader['y_test'] 为空")
        return None

    realy = torch.tensor(y_test, dtype=torch.float32, device=device)
    realy = realy.transpose(1, 3)[:, 0, :, :]  # [num_samples, num_nodes, seq_length]

    # 预统计batch数
    test_iter = test_loader.get_iterator()
    test_batches = 0
    for _ in test_iter:
        test_batches += 1

    # 重新获取迭代器
    test_iter = test_loader.get_iterator()

    # 进行推理
    outputs = []
    bt0 = time.time()

    with torch.no_grad():
        for it, (x, y) in enumerate(test_iter):
            testx = torch.tensor(x, dtype=torch.float32, device=device)
            testx = testx.permute(0, 3, 2, 1)  # [B, in_dim, N, seq_in]

            preds = model(testx)  # Model output: [B, out_dim, N, 1]
            preds = preds.permute(0, 3, 2, 1)  # [B, 1, N, out_dim]

            # Debug: print model output shape
            if it == 0:
                log(f"DEBUG: Model output shape: {preds.shape} (before squeeze)")
                log(f"DEBUG: After squeeze(1): {preds.squeeze(1).shape}")

            outputs.append(preds.squeeze(1))  # Remove dimension 1 only -> [B, N, out_dim]

            if (it % max(1, args.print_every) == 0):
                log(f"Batch {it}/{test_batches} processed.")

    yhatt = torch.cat(outputs, dim=0)  # [num_samples, num_nodes, seq_length]

    # 对齐长度
    if yhatt.size(0) < realy.size(0):
        realy_eval = realy[:yhatt.size(0), ...]
    else:
        realy_eval = realy
        yhatt = yhatt[:realy_eval.size(0), ...]

    log(f"Inference completed in {time.time()-bt0:.2f}s, Shape={yhatt.shape}")

    # 计算每个时间步的指标
    log("\n" + "="*80)
    log("Testing Results")
    log("="*80)

    mae_list = []
    rmse_list = []
    mape_list = []
    crps_list = []
    wasserstein_list = []
    predictions = []
    ground_truth = []

    for h in range(args.seq_length):
        # 反归一化预测结果
        # yhatt[:, :, h] -> (num_samples, num_nodes)
        pred_normalized = yhatt[:, :, h]  # [num_samples, num_nodes]
        pred = scaler.inverse_transform(pred_normalized)
        real = realy_eval[:, :, h]  # y_test 本身就是原始尺度，不需要反归一化

        # Debug: print shapes
        if h == 0:
            log(f"DEBUG: yhatt shape: {yhatt.shape}")
            log(f"DEBUG: realy_eval shape: {realy_eval.shape}")
            log(f"DEBUG: pred_normalized shape: {pred_normalized.shape}")
            log(f"DEBUG: pred shape after inverse_transform: {pred.shape}")
            log(f"DEBUG: real shape: {real.shape}")
            log(f"DEBUG: scaler.mean type: {type(scaler.mean)}, scaler.std type: {type(scaler.std)}")
            if hasattr(scaler.mean, 'shape'):
                log(f"DEBUG: scaler.mean shape: {scaler.mean.shape if hasattr(scaler.mean, 'shape') else 'scalar'}")
                log(f"DEBUG: scaler.std shape: {scaler.std.shape if hasattr(scaler.std, 'shape') else 'scalar'}")
            else:
                log(f"DEBUG: scaler.mean value: {scaler.mean}, scaler.std value: {scaler.std}")

        # 计算传统指标（使用原始尺度）
        mae, mape, rmse = util.metric(pred, real)
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)

        # 计算新指标（使用原始尺度）
        crps = compute_crps(pred, real)
        wd = compute_wasserstein(pred, real)
        crps_list.append(crps)
        wasserstein_list.append(wd)

        if h % 12 == 0 or h == args.seq_length - 1:
            log(f'Horizon {h+1:3d} | MAE: {mae:.4f} | MAPE: {mape:.4f} | RMSE: {rmse:.4f} | CRPS: {crps:.4f} | WD: {wd:.4f}')

        # 保存用于可视化 (num_samples, num_nodes)
        predictions.append(pred.cpu().numpy())
        ground_truth.append(real.cpu().numpy())

    log("="*80)
    log(f'Average | MAE: {np.mean(mae_list):.4f} | MAPE: {np.mean(mape_list):.4f} | RMSE: {np.mean(rmse_list):.4f} | CRPS: {np.mean(crps_list):.4f} | WD: {np.mean(wasserstein_list):.4f}')
    log("="*80 + "\n")

    # 转换为numpy数组: (seq_length, num_samples, num_nodes)
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    return predictions, ground_truth, mae_list, mape_list, rmse_list, crps_list, wasserstein_list


def plot_results(predictions, ground_truth, mae_list, mape_list, rmse_list, crps_list, wasserstein_list, args, save_dir):
    """可视化预测结果 (STGCN风格 + 新指标)"""
    os.makedirs(save_dir, exist_ok=True)
    log(f"\nGenerating visualizations in {save_dir}...")

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    seq_length = args.seq_length

    # 1. 绘制所有指标的性能曲线（5个指标）
    log("Plotting metrics over horizons...")
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    horizons = list(range(1, seq_length + 1))

    # MAE
    axes[0, 0].plot(horizons, mae_list, marker='o', linewidth=2, markersize=4, markevery=12)
    axes[0, 0].set_xlabel('Prediction Horizon', fontsize=12)
    axes[0, 0].set_ylabel('MAE', fontsize=12)
    axes[0, 0].set_title('MAE vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(12))

    # MAPE
    axes[0, 1].plot(horizons, mape_list, marker='s', linewidth=2, markersize=4, markevery=12, color='orange')
    axes[0, 1].set_xlabel('Prediction Horizon', fontsize=12)
    axes[0, 1].set_ylabel('MAPE (%)', fontsize=12)
    axes[0, 1].set_title('MAPE vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(12))

    # RMSE
    axes[0, 2].plot(horizons, rmse_list, marker='^', linewidth=2, markersize=4, markevery=12, color='green')
    axes[0, 2].set_xlabel('Prediction Horizon', fontsize=12)
    axes[0, 2].set_ylabel('RMSE', fontsize=12)
    axes[0, 2].set_title('RMSE vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].xaxis.set_major_locator(ticker.MultipleLocator(12))

    # CRPS
    axes[1, 0].plot(horizons, crps_list, marker='D', linewidth=2, markersize=4, markevery=12, color='purple')
    axes[1, 0].set_xlabel('Prediction Horizon', fontsize=12)
    axes[1, 0].set_ylabel('CRPS', fontsize=12)
    axes[1, 0].set_title('CRPS vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(12))

    # Wasserstein Distance
    axes[1, 1].plot(horizons, wasserstein_list, marker='*', linewidth=2, markersize=6, markevery=12, color='red')
    axes[1, 1].set_xlabel('Prediction Horizon', fontsize=12)
    axes[1, 1].set_ylabel('Wasserstein Distance', fontsize=12)
    axes[1, 1].set_title('Wasserstein Distance vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(12))

    # 综合对比（归一化后）
    mae_norm = np.array(mae_list) / np.max(mae_list)
    mape_norm = np.array(mape_list) / np.max(mape_list)
    rmse_norm = np.array(rmse_list) / np.max(rmse_list)
    crps_norm = np.array(crps_list) / np.max(crps_list)
    wd_norm = np.array(wasserstein_list) / np.max(wasserstein_list)

    axes[1, 2].plot(horizons, mae_norm, label='MAE', linewidth=2, alpha=0.7)
    axes[1, 2].plot(horizons, mape_norm, label='MAPE', linewidth=2, alpha=0.7)
    axes[1, 2].plot(horizons, rmse_norm, label='RMSE', linewidth=2, alpha=0.7)
    axes[1, 2].plot(horizons, crps_norm, label='CRPS', linewidth=2, alpha=0.7)
    axes[1, 2].plot(horizons, wd_norm, label='WD', linewidth=2, alpha=0.7)
    axes[1, 2].set_xlabel('Prediction Horizon', fontsize=12)
    axes[1, 2].set_ylabel('Normalized Metric', fontsize=12)
    axes[1, 2].set_title('All Metrics Comparison (Normalized)', fontsize=14, fontweight='bold')
    axes[1, 2].legend(fontsize=10, loc='best')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].xaxis.set_major_locator(ticker.MultipleLocator(12))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_over_horizons.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Saved: metrics_over_horizons.png")

    # 2. 绘制预测vs真实值的散点图
    log("Plotting prediction vs ground truth scatter plots...")
    selected_horizons = [0, seq_length//4, seq_length//2, seq_length-1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, h in enumerate(selected_horizons):
        pred_h = predictions[h].flatten()
        real_h = ground_truth[h].flatten()

        # 随机采样
        sample_size = min(10000, len(pred_h))
        indices = np.random.choice(len(pred_h), sample_size, replace=False)
        pred_sample = pred_h[indices]
        real_sample = real_h[indices]

        axes[idx].scatter(real_sample, pred_sample, alpha=0.3, s=10)
        axes[idx].plot([real_sample.min(), real_sample.max()],
                       [real_sample.min(), real_sample.max()],
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[idx].set_xlabel('Ground Truth', fontsize=11)
        axes[idx].set_ylabel('Prediction', fontsize=11)
        axes[idx].set_title(f'Horizon {h+1} (MAE: {mae_list[h]:.4f}, WD: {wasserstein_list[h]:.4f})',
                           fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_vs_groundtruth.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Saved: prediction_vs_groundtruth.png")

    # 3. 绘制时间序列预测示例（2行2列，每个节点一个样本）
    log("Plotting time series predictions...")

    # 解析要绘制的节点列表
    if args.plot_nodes is not None:
        try:
            plot_node_list = [int(n.strip()) for n in args.plot_nodes.split(',')]
        except:
            log(f"WARNING: Invalid --plot_nodes format, using default [0, 1, 2, 3]")
            plot_node_list = [0, 1, 2, 3]
    else:
        # 默认选择前4个节点
        plot_node_list = [0, 1, 2, 3]

    # 限制为最多4个节点（2x2布局）
    plot_node_list = plot_node_list[:4]

    # 检查节点索引是否有效
    max_node_idx = predictions.shape[2] - 1
    plot_node_list = [n for n in plot_node_list if 0 <= n <= max_node_idx]

    if len(plot_node_list) == 0:
        log("WARNING: No valid nodes to plot")
    else:
        # 检查样本索引是否有效
        sample_idx = args.plot_sample_idx
        max_sample_idx = predictions.shape[1] - 1
        if sample_idx < 0 or sample_idx > max_sample_idx:
            log(f"WARNING: Invalid sample index {sample_idx}, using 0")
            sample_idx = 0

        log(f"Plotting nodes: {plot_node_list}, sample: {sample_idx}")

        # 创建2行2列的子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        for idx, node_idx in enumerate(plot_node_list):
            pred_series = predictions[:, sample_idx, node_idx]
            real_series = ground_truth[:, sample_idx, node_idx]

            axes[idx].plot(range(1, seq_length+1), real_series,
                          label='Ground Truth', linewidth=2, color='#005f73', alpha=0.9)
            axes[idx].plot(range(1, seq_length+1), pred_series,
                          label='Prediction', linewidth=2, linestyle='--', color='#ae2012', alpha=0.9)
            axes[idx].set_xlabel('Time Step', fontsize=11)
            axes[idx].set_ylabel('Traffic Speed', fontsize=11)
            axes[idx].set_title(f'Node {node_idx}, Sample {sample_idx}',
                               fontsize=12, fontweight='bold')
            axes[idx].legend(fontsize=9, frameon=False, loc='best')
            axes[idx].grid(False)
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)
            axes[idx].xaxis.set_major_locator(ticker.MultipleLocator(12))

        # 如果节点少于4个，隐藏多余的子图
        for idx in range(len(plot_node_list), 4):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'time_series_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        log(f"Saved: time_series_predictions.png")

    # 4. 绘制误差分布直方图
    log("Plotting error distribution...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, h in enumerate(selected_horizons):
        errors = predictions[h].flatten() - ground_truth[h].flatten()

        axes[idx].hist(errors, bins=100, alpha=0.7, edgecolor='black')
        axes[idx].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[idx].set_xlabel('Prediction Error', fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f'Error Distribution - Horizon {h+1}\n(Mean: {errors.mean():.4f}, Std: {errors.std():.4f})',
                           fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Saved: error_distribution.png")

    # 5. 绘制空间误差热力图
    log("Plotting spatial error heatmap...")
    node_mae = np.mean(np.abs(predictions - ground_truth), axis=(0, 1))  # (num_nodes,)

    fig, ax = plt.subplots(figsize=(12, 8))

    n_nodes = len(node_mae)
    grid_size = int(np.ceil(np.sqrt(n_nodes)))
    node_mae_grid = np.zeros((grid_size, grid_size))
    node_mae_grid.flat[:n_nodes] = node_mae

    im = ax.imshow(node_mae_grid, cmap='YlOrRd', aspect='auto')
    ax.set_title('Average MAE per Node (Spatial Distribution)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Node Grid X', fontsize=12)
    ax.set_ylabel('Node Grid Y', fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MAE', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spatial_error_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    log(f"Saved: spatial_error_heatmap.png")

    log(f"\nAll plots saved to: {save_dir}")


def save_results(predictions, ground_truth, mae_list, mape_list, rmse_list, crps_list, wasserstein_list, save_dir):
    """保存预测结果和指标"""
    np.save(os.path.join(save_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(save_dir, 'ground_truth.npy'), ground_truth)

    # 保存所有指标
    metrics = {
        'mae': mae_list,
        'mape': mape_list,
        'rmse': rmse_list,
        'crps': crps_list,
        'wasserstein': wasserstein_list
    }
    np.save(os.path.join(save_dir, 'metrics.npy'), metrics)

    # 保存指标摘要
    with open(os.path.join(save_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("Metrics Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"MAE:               {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}\n")
        f.write(f"MAPE:              {np.mean(mape_list):.4f} ± {np.std(mape_list):.4f}\n")
        f.write(f"RMSE:              {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}\n")
        f.write(f"CRPS:              {np.mean(crps_list):.4f} ± {np.std(crps_list):.4f}\n")
        f.write(f"Wasserstein Dist:  {np.mean(wasserstein_list):.4f} ± {np.std(wasserstein_list):.4f}\n")
        f.write("\n" + "="*60 + "\n")

    log(f"Results saved to: {save_dir}")


def main():
    t0 = time.time()
    log("="*80)
    log("Model Testing with STGCN-Style Visualization + Advanced Metrics")
    log("="*80)
    log(f"Command: {' '.join(sys.argv)}")

    args = parse_args()
    device = set_device(args.device)

    log(f"\n{'='*80}")
    log("Configuration")
    log("="*80)
    if args.dataset:
        log(f"Dataset: {args.dataset.upper()}")
    log(f"Device: {device}")
    log(f"Model: {args.model}")
    log(f"Checkpoint: {args.ckpt}")
    log(f"Data: {args.data}")
    log(f"Adjacency: {args.adjdata}")
    log(f"Num Nodes: {args.num_nodes}")
    log(f"Sequence Length: {args.seq_length}")

    # 动态导入模型模块
    try:
        model_module = importlib.import_module(f'models.{args.model}')
        log(f'Successfully loaded model: models.{args.model}')
    except ImportError:
        log(f'Model models.{args.model} not found, falling back to default model_cfg')
        import model_cfg as model_module

    if not os.path.isfile(args.ckpt):
        log(f"ERROR: Checkpoint file not found: {args.ckpt}")
        return

    try:
        # 加载邻接矩阵
        log("\nLoading adjacency matrix...")
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
        supports = model_module.make_supports(adj_mx, device)
        log("Adjacency matrix loaded")

        # 加载数据（使用test_all_fig.py的方式）
        log("\nLoading dataset...")
        dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
        scaler = dataloader['scaler']
        log("Dataset loaded successfully")

        # 构建和加载模型
        log("\nBuilding model...")
        model = build_model(args, device, supports, model_module)

        log("Loading checkpoint...")
        state = torch.load(args.ckpt, map_location=device)
        try:
            model.load_state_dict(state, strict=True)
            log("Model loaded (strict=True)")
        except Exception as e:
            log(f"WARNING: Strict loading failed, trying strict=False. Error: {str(e)[:100]}...")
            model.load_state_dict(state, strict=False)
            log("Model loaded (strict=False)")

        model.eval()
        log("Model ready for inference")

        # 测试模型
        log("\n" + "="*80)
        log("Starting inference...")
        log("="*80)
        result = test_model(args, model, dataloader, scaler, device)

        if result is None:
            log("ERROR: Testing failed")
            return

        predictions, ground_truth, mae_list, mape_list, rmse_list, crps_list, wasserstein_list = result

        # 创建保存目录
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f"{args.fig_dir}_{current_time}"
        os.makedirs(save_dir, exist_ok=True)

        # 可视化结果
        plot_results(predictions, ground_truth, mae_list, mape_list, rmse_list,
                    crps_list, wasserstein_list, args, save_dir)

        # 保存结果
        save_results(predictions, ground_truth, mae_list, mape_list, rmse_list,
                    crps_list, wasserstein_list, save_dir)

        log("\n" + "="*80)
        log(f"Testing and visualization completed successfully!")
        log(f"Total time: {time.time()-t0:.2f}s")
        log("="*80)

    except Exception as e:
        log("\nFATAL ERROR:")
        traceback.print_exc()


if __name__ == '__main__':
    main()
