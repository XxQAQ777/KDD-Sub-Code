import sys
sys.path.append('../')

import os
import argparse
import configparser
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import wasserstein_distance
from lib.utils import *
from lib.data_loader import *
from lib.generate_adj_mx import *
from STGCN_Utils import *
from model.STGCN.stgcn import STGCN as Network


def compute_crps(pred, real):
    """
    计算CRPS (Continuous Ranked Probability Score)
    对于确定性预测，CRPS简化为MAE
    pred, real: torch tensors of shape (num_samples, num_nodes, 1)
    """
    # 对于确定性预测，CRPS = MAE
    crps = torch.mean(torch.abs(pred - real)).item()
    return crps


def compute_wasserstein(pred, real):
    """
    计算Wasserstein Distance (Earth Mover's Distance)
    pred, real: torch tensors of shape (num_samples, num_nodes, 1)
    """
    pred_np = pred.cpu().numpy().flatten()
    real_np = real.cpu().numpy().flatten()

    # 使用scipy的wasserstein_distance
    # 这计算的是1-Wasserstein距离
    wd = wasserstein_distance(pred_np, real_np)
    return wd


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='STGCN Model Testing and Visualization')

    # 数据集选择
    parser.add_argument('--dataset', type=str, default='PEMSBAY',
                        choices=['METRLA', 'PEMSBAY'],
                        help='Dataset to use: METRLA or PEMSBAY')

    # 模型路径
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model checkpoint. If not specified, will use default path.')

    # 设备
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use: cuda:0, cuda:1, or cpu')

    # 测试参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for testing')

    args = parser.parse_args()
    return args


def setup_dataset_config(dataset_name):
    """
    根据数据集名称从配置文件读取配置

    Returns:
        dict: 包含数据集配置的字典
    """
    # 基本路径配置
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
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(base_configs.keys())}")

    base_config = base_configs[dataset_name]

    # 读取配置文件
    config_file = base_config['config_file']
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config = configparser.ConfigParser()
    config.read(config_file)

    # 从配置文件读取参数
    full_config = base_config.copy()
    full_config['num_nodes'] = int(config['data']['num_nodes'])
    full_config['window'] = int(config['data']['window'])
    full_config['horizon'] = int(config['data']['horizon'])
    full_config['input_dim'] = eval(config['model']['input_dim'])
    full_config['KS'] = eval(config['model']['KS'])
    full_config['KT'] = eval(config['model']['KT'])
    full_config['channels'] = eval(config['model']['channels'])  # 保留原始格式
    full_config['dropout'] = eval(config['model']['dropout'])

    return full_config


def find_latest_model(pattern):
    """
    根据模式查找最新的模型文件

    Args:
        pattern: 模型路径模式（可以包含通配符*）

    Returns:
        str: 最新的模型文件路径
    """
    import glob

    # 如果pattern包含通配符，查找所有匹配的文件
    if '*' in pattern:
        matching_files = glob.glob(pattern)
        if not matching_files:
            return None
        # 按修改时间排序，返回最新的
        matching_files.sort(key=os.path.getmtime, reverse=True)
        return matching_files[0]
    else:
        return pattern if os.path.exists(pattern) else None


class SimpleArgs:
    """简单的参数类，用于替代argparse.Namespace"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def load_data(args):
    """加载数据集和邻接矩阵"""
    data_loader = load_dataset(args.dataset_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler = data_loader['scaler']
    # 加载拓扑图的邻接矩阵
    _, _, adj_mx = load_pickle(args.graph_pkl)
    adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    return adj_mx, data_loader, scaler


def generate_model(args, adj_mx):
    """生成模型"""
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
    print_model_parameters(model, only_num=False)
    return model


def test_model(args, model, data_loader, scaler, model_path):
    """测试模型并返回预测结果和真实值"""
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    model.eval()
    print(f"Loaded model from: {model_path}")

    # 准备测试数据
    test_loader = data_loader['test_loader']
    realy = torch.Tensor(data_loader['y_test']).to(args.device)
    realy = realy[:, :, :, 0:1]  # (B, T, N, 1)

    # 进行预测
    outputs = []
    with torch.no_grad():
        for _, (x, y) in enumerate(test_loader.get_iterator()):
            testx = torch.Tensor(x).to(args.device)
            output = model(testx[:, :, :, :args.input_dim])
            outputs.append(output)

    # 合并所有批次的预测结果
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    # 计算每个时间步的指标
    print("\n" + "="*80)
    print("Testing Results on Test Set")
    print("="*80)

    mae_list = []
    rmse_list = []
    mape_list = []
    crps_list = []
    wasserstein_list = []

    predictions = []
    ground_truth = []

    for i in range(args.horizon):
        # 反归一化预测结果
        pred = scaler.inverse_transform(yhat[:, i, :, :])
        real = realy[:, i, :, :]

        # 计算传统指标
        metrics = metric(pred, real)
        mae_list.append(metrics[0])
        mape_list.append(metrics[1])
        rmse_list.append(metrics[2])

        # 计算新指标
        crps = compute_crps(pred, real)
        wd = compute_wasserstein(pred, real)
        crps_list.append(crps)
        wasserstein_list.append(wd)

        if i % 3 == 0 or i == args.horizon - 1:
            print(f'Horizon {i+1:3d} | MAE: {metrics[0]:.4f} | MAPE: {metrics[1]:.4f} | RMSE: {metrics[2]:.4f} | CRPS: {crps:.4f} | WD: {wd:.4f}')

        # 保存预测和真实值用于可视化
        predictions.append(pred.cpu().numpy())
        ground_truth.append(real.cpu().numpy())

    print("="*80)
    print(f'Average over {args.horizon} horizons | MAE: {np.mean(mae_list):.4f} | MAPE: {np.mean(mape_list):.4f} | RMSE: {np.mean(rmse_list):.4f} | CRPS: {np.mean(crps_list):.4f} | WD: {np.mean(wasserstein_list):.4f}')
    print("="*80 + "\n")

    # 转换为numpy数组
    predictions = np.array(predictions)  # (T, B, N, 1)
    ground_truth = np.array(ground_truth)  # (T, B, N, 1)

    return predictions, ground_truth, mae_list, mape_list, rmse_list, crps_list, wasserstein_list


def plot_results(predictions, ground_truth, mae_list, mape_list, rmse_list, crps_list, wasserstein_list, args, save_dir):
    """可视化预测结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 1. 绘制不同时间步的性能指标（现在有5个指标）
    print("Plotting metrics over horizons...")
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    horizons = list(range(1, args.horizon + 1))

    # MAE
    axes[0, 0].plot(horizons, mae_list, marker='o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Prediction Horizon', fontsize=12)
    axes[0, 0].set_ylabel('MAE', fontsize=12)
    axes[0, 0].set_title('MAE vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # MAPE
    axes[0, 1].plot(horizons, mape_list, marker='s', linewidth=2, markersize=6, color='orange')
    axes[0, 1].set_xlabel('Prediction Horizon', fontsize=12)
    axes[0, 1].set_ylabel('MAPE (%)', fontsize=12)
    axes[0, 1].set_title('MAPE vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # RMSE
    axes[0, 2].plot(horizons, rmse_list, marker='^', linewidth=2, markersize=6, color='green')
    axes[0, 2].set_xlabel('Prediction Horizon', fontsize=12)
    axes[0, 2].set_ylabel('RMSE', fontsize=12)
    axes[0, 2].set_title('RMSE vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)

    # CRPS
    axes[1, 0].plot(horizons, crps_list, marker='D', linewidth=2, markersize=6, color='purple')
    axes[1, 0].set_xlabel('Prediction Horizon', fontsize=12)
    axes[1, 0].set_ylabel('CRPS', fontsize=12)
    axes[1, 0].set_title('CRPS vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Wasserstein Distance
    axes[1, 1].plot(horizons, wasserstein_list, marker='*', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Prediction Horizon', fontsize=12)
    axes[1, 1].set_ylabel('Wasserstein Distance', fontsize=12)
    axes[1, 1].set_title('Wasserstein Distance vs Prediction Horizon', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

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

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_over_horizons.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'metrics_over_horizons.png')}")

    # 2. 绘制预测vs真实值的散点图（选择几个时间步）
    print("Plotting prediction vs ground truth scatter plots...")
    selected_horizons = [0, args.horizon//4, args.horizon//2, args.horizon-1]  # 选择4个时间步
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, h in enumerate(selected_horizons):
        pred_h = predictions[h].flatten()
        real_h = ground_truth[h].flatten()

        # 随机采样以加快绘图速度
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
    print(f"Saved: {os.path.join(save_dir, 'prediction_vs_groundtruth.png')}")

    # 3. 绘制时间序列预测示例（选择几个节点）
    print("Plotting time series predictions for sample nodes...")
    num_samples = min(5, predictions.shape[1])  # 选择5个样本
    num_nodes_to_plot = min(3, predictions.shape[2])  # 选择3个节点

    fig, axes = plt.subplots(num_nodes_to_plot, num_samples, figsize=(20, 10))
    if num_nodes_to_plot == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for node_idx in range(num_nodes_to_plot):
        for sample_idx in range(num_samples):
            pred_series = predictions[:, sample_idx, node_idx, 0]
            real_series = ground_truth[:, sample_idx, node_idx, 0]

            axes[node_idx, sample_idx].plot(range(1, args.horizon+1), real_series,
                                           label='Ground Truth', linewidth=2, marker='o', markersize=4)
            axes[node_idx, sample_idx].plot(range(1, args.horizon+1), pred_series,
                                           label='Prediction', linewidth=2, marker='s', markersize=4)
            axes[node_idx, sample_idx].set_xlabel('Time Step', fontsize=10)
            axes[node_idx, sample_idx].set_ylabel('Speed', fontsize=10)
            axes[node_idx, sample_idx].set_title(f'Node {node_idx+1}, Sample {sample_idx+1}',
                                                fontsize=11, fontweight='bold')
            axes[node_idx, sample_idx].legend(fontsize=8)
            axes[node_idx, sample_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'time_series_predictions.png')}")

    # 4. 绘制误差分布直方图
    print("Plotting error distribution...")
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
    print(f"Saved: {os.path.join(save_dir, 'error_distribution.png')}")

    # 5. 绘制空间误差热力图（平均每个节点的误差）
    print("Plotting spatial error heatmap...")
    # 计算每个节点在所有时间步和样本上的平均MAE
    node_mae = np.mean(np.abs(predictions - ground_truth), axis=(0, 1, 3))  # (N,)

    fig, ax = plt.subplots(figsize=(12, 8))

    # 将节点MAE重塑为2D以便可视化（如果节点数太多，可以调整）
    n_nodes = len(node_mae)
    grid_size = int(np.ceil(np.sqrt(n_nodes)))
    node_mae_grid = np.zeros((grid_size, grid_size))
    node_mae_grid.flat[:n_nodes] = node_mae

    im = ax.imshow(node_mae_grid, cmap='YlOrRd', aspect='auto')
    ax.set_title('Average MAE per Node (Spatial Distribution)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Node Grid X', fontsize=12)
    ax.set_ylabel('Node Grid Y', fontsize=12)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MAE', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spatial_error_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'spatial_error_heatmap.png')}")

    print(f"\nAll plots saved to: {save_dir}")


def save_predictions(predictions, ground_truth, save_dir):
    """保存预测结果和真实值"""
    np.save(os.path.join(save_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(save_dir, 'ground_truth.npy'), ground_truth)
    print(f"\nPredictions and ground truth saved to: {save_dir}")


def save_metrics_to_csv(mae_list, mape_list, rmse_list, crps_list, wasserstein_list, args, save_dir):
    """保存所有指标到CSV文件"""
    # 创建DataFrame
    metrics_df = pd.DataFrame({
        'Horizon': list(range(1, args.horizon + 1)),
        'MAE': mae_list,
        'MAPE': mape_list,
        'RMSE': rmse_list,
        'CRPS': crps_list,
        'Wasserstein_Distance': wasserstein_list
    })

    # 保存详细指标
    csv_path = os.path.join(save_dir, 'metrics_detailed.csv')
    metrics_df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Detailed metrics saved to: {csv_path}")

    # 保存汇总统计
    summary_df = pd.DataFrame({
        'Metric': ['MAE', 'MAPE', 'RMSE', 'CRPS', 'Wasserstein_Distance'],
        'Mean': [
            np.mean(mae_list),
            np.mean(mape_list),
            np.mean(rmse_list),
            np.mean(crps_list),
            np.mean(wasserstein_list)
        ],
        'Std': [
            np.std(mae_list),
            np.std(mape_list),
            np.std(rmse_list),
            np.std(crps_list),
            np.std(wasserstein_list)
        ],
        'Min': [
            np.min(mae_list),
            np.min(mape_list),
            np.min(rmse_list),
            np.min(crps_list),
            np.min(wasserstein_list)
        ],
        'Max': [
            np.max(mae_list),
            np.max(mape_list),
            np.max(rmse_list),
            np.max(crps_list),
            np.max(wasserstein_list)
        ]
    })

    summary_csv_path = os.path.join(save_dir, 'metrics_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f')
    print(f"Summary metrics saved to: {summary_csv_path}")

    # 同时保存为文本文件以便阅读
    summary_txt_path = os.path.join(save_dir, 'metrics_summary.txt')
    with open(summary_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Metrics Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"MAE:                {np.mean(mae_list):.6f} ± {np.std(mae_list):.6f}\n")
        f.write(f"MAPE:               {np.mean(mape_list):.6f} ± {np.std(mape_list):.6f}\n")
        f.write(f"RMSE:               {np.mean(rmse_list):.6f} ± {np.std(rmse_list):.6f}\n")
        f.write(f"CRPS:               {np.mean(crps_list):.6f} ± {np.std(crps_list):.6f}\n")
        f.write(f"Wasserstein Dist:   {np.mean(wasserstein_list):.6f} ± {np.std(wasserstein_list):.6f}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("\nDetailed Statistics:\n")
        f.write("="*80 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n" + "="*80 + "\n")
    print(f"Summary text saved to: {summary_txt_path}")


if __name__ == '__main__':
    # 解析命令行参数
    cmd_args = parse_args()

    # 设置数据集配置
    dataset_config = setup_dataset_config(cmd_args.dataset)

    # 创建统一的args对象
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

    # 设置设备
    if torch.cuda.is_available() and args.device.startswith('cuda'):
        device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        torch.cuda.set_device(device_id)
    else:
        args.device = 'cpu'

    print("="*80)
    print("STGCN Model Testing and Visualization")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Num Nodes: {args.num_nodes}")
    print(f"Window: {args.window}")
    print(f"Horizon: {args.horizon}")
    print(f"Dataset Dir: {args.dataset_dir}")
    print(f"Graph Pkl: {args.graph_pkl}")
    print("="*80 + "\n")

    # 加载数据
    print("Loading data...")
    adj_mx, data_loader, scaler = load_data(args)
    print("Data loaded successfully!\n")

    # 生成模型
    print("Generating model...")
    model = generate_model(args, adj_mx)
    print("Model generated successfully!\n")

    # 确定模型路径
    if cmd_args.model_path is not None:
        model_path = cmd_args.model_path
    else:
        # 使用默认路径模式查找最新模型
        model_path = find_latest_model(dataset_config['default_model_path'])

    if model_path is None or not os.path.exists(model_path):
        print(f"Error: Model file not found!")
        if cmd_args.model_path is not None:
            print(f"  Specified path: {cmd_args.model_path}")
        else:
            print(f"  Searched pattern: {dataset_config['default_model_path']}")
        print("\nPlease specify the model path using --model_path argument.")
        print("Example:")
        print(f"  python test_and_plot.py --dataset {args.dataset} --model_path /path/to/model.pth")
        sys.exit(1)

    print(f"Using model: {model_path}\n")

    # 测试模型
    print("Testing model...")
    predictions, ground_truth, mae_list, mape_list, rmse_list, crps_list, wasserstein_list = test_model(
        args, model, data_loader, scaler, model_path
    )

    # 创建保存目录
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"./test_results_{args.dataset}_{current_time}"
    os.makedirs(save_dir, exist_ok=True)

    # 可视化结果
    print("\nGenerating visualizations...")
    plot_results(predictions, ground_truth, mae_list, mape_list, rmse_list,
                crps_list, wasserstein_list, args, save_dir)

    # 保存预测结果
    save_predictions(predictions, ground_truth, save_dir)

    # 保存指标到CSV
    print("\nSaving metrics to CSV...")
    save_metrics_to_csv(mae_list, mape_list, rmse_list, crps_list, wasserstein_list, args, save_dir)

    print("\n" + "="*80)
    print("Testing and visualization completed successfully!")
    print(f"Results saved to: {save_dir}")
    print("="*80)
