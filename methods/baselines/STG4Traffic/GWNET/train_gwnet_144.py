#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GWNET 144->144时间步训练脚本
使用方法: python train_gwnet_144.py
"""

import sys
sys.path.append('../')

import os
import torch
from datetime import datetime
from lib.utils import *
from lib.data_loader import *
from lib.generate_adj_mx import *
from GWNET_Config_144 import args
from GWNET_Trainer import Trainer
from model.GWNET.gwnet import gwnet as Network


def load_data(args):
    """加载数据和邻接矩阵"""
    print("=" * 80)
    print("Loading dataset...")
    print(f"Dataset directory: {args.dataset_dir}")

    data_loader = load_dataset(args.dataset_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler = data_loader['scaler']

    # 加载拓扑图的邻接矩阵
    print(f"Loading adjacency matrix from: {args.graph_pkl}")
    _, _, adj_mx = load_pickle(args.graph_pkl)
    adj_mx = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]

    print("Data loaded successfully!")
    print(f"Train samples: {data_loader['x_train'].shape[0]}")
    print(f"Val samples: {data_loader['x_val'].shape[0]}")
    print(f"Test samples: {data_loader['x_test'].shape[0]}")
    print(f"Input shape: {data_loader['x_train'].shape}")
    print(f"Output shape: {data_loader['y_train'].shape}")
    print("=" * 80)

    return adj_mx, data_loader, scaler


def generate_model_components(args, supports):
    """生成模型、损失函数、优化器和学习率调度器"""
    print("=" * 80)
    print("Initializing model...")

    # 如果使用多GPU，将supports放到主设备上
    device = torch.device(args.device.split(',')[0] if ',' in args.device else args.device)
    supports = [torch.tensor(adj).to(device) for adj in supports]

    # 1. 模型
    model = Network(
        device,
        num_nodes=args.num_nodes,
        dropout=args.dropout,
        supports=supports,
        gcn_bool=True,
        addaptadj=True,
        in_dim=args.input_dim,
        out_dim=args.horizon,  # 144个时间步输出
        residual_channels=args.hidden_dim,
        dilation_channels=args.hidden_dim,
        skip_channels=8 * args.hidden_dim,
        end_channels=16 * args.hidden_dim,
        blocks=args.blocks,
        layers=args.layers
    )
    model = model.to(device)

    # 如果有多个GPU，使用DataParallel
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {len(args.gpu_ids)} GPUs: {args.gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        print("Model wrapped with DataParallel")

    # 打印模型参数
    print_model_parameters(model, only_num=False)

    # 2. 损失函数
    if args.loss_func == 'mask_mae':
        loss = masked_mae
    elif args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(device)
    elif args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(device)
    elif args.loss_func == 'smoothloss':
        loss = torch.nn.SmoothL1Loss().to(device)
    else:
        raise ValueError(f"Unknown loss function: {args.loss_func}")

    # 3. 优化器
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=args.weight_decay,
        amsgrad=False
    )

    # 4. 学习率衰减
    lr_scheduler = None
    if args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=lr_decay_steps,
            gamma=args.lr_decay_rate
        )

    print("Model initialized successfully!")
    print("=" * 80)

    return model, loss, optimizer, lr_scheduler


def get_log_dir(model, dataset):
    """获取日志目录"""
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
    log_dir = os.path.join(current_dir, 'log', model, dataset + '_144', current_time)
    return log_dir


def main():
    print("\n" + "=" * 80)
    print("GWNET 144->144 时间步预测训练")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset}")
    print(f"Input window: {args.window}")
    print(f"Prediction horizon: {args.horizon}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr_init}")
    print(f"Curriculum learning: {args.cl}")
    print(f"Model blocks: {args.blocks}")
    print(f"Model layers: {args.layers}")
    print("=" * 80 + "\n")

    # 设置设备
    if torch.cuda.is_available():
        # 支持多GPU设置
        if hasattr(args, 'use_multi_gpu') and args.use_multi_gpu:
            # 多GPU模式
            print(f"Available GPUs: {torch.cuda.device_count()}")
            if torch.cuda.device_count() >= 2:
                args.gpu_ids = [0, 1]  # 使用GPU 0和1
                args.device = 'cuda:0'  # 主设备
                torch.cuda.set_device(0)
                print(f"Using multi-GPU mode on: {args.gpu_ids}")
                for gpu_id in args.gpu_ids:
                    print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            else:
                print("Warning: Multi-GPU mode requested but only 1 GPU available")
                args.use_multi_gpu = False
                args.gpu_ids = [0]
                args.device = 'cuda:0'
                torch.cuda.set_device(0)
                print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            # 单GPU模式
            device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
            torch.cuda.set_device(device_id)
            print(f"Using single GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
    else:
        args.device = 'cpu'
        args.use_multi_gpu = False
        print("CUDA not available, using CPU")

    # 加载数据
    supports, data_loader, scaler = load_data(args)

    # 设置日志目录
    args.log_dir = get_log_dir(args.model, args.dataset)
    print(f"Log directory: {args.log_dir}\n")

    # 生成模型组件
    model, loss, optimizer, lr_scheduler = generate_model_components(args, supports)

    # 创建训练器
    trainer = Trainer(
        args=args,
        data_loader=data_loader,
        scaler=scaler,
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        cl=args.cl,
        new_training_method=args.new_training_method
    )

    # 开始训练
    if args.mode == "train":
        print("\nStarting training...\n")
        trainer.train()
        print("\nTraining completed!")
    elif args.mode == 'test':
        # 指定checkpoint路径
        checkpoint = "../log/GWNET/METRLA_144/20260104071818/METRLA_GWNET_best_model.pth"
        if not os.path.exists(checkpoint):
            print(f"Error: Checkpoint not found at {checkpoint}")
            print("Please specify a valid checkpoint path in GWNET_Main_144.py")
            return
        print(f"\nLoading checkpoint from: {checkpoint}")
        trainer.test(args, model, data_loader, scaler, trainer.logger, save_path=checkpoint)
        print("\nTesting completed!")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
