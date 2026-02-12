import argparse
import numpy as np
import os
import torch
import datetime
import time
import yaml
import sys
import matplotlib.pyplot as plt

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


def test_checkpoint(
    checkpoint_path,
    dataset="METRLA",
    gpu_num=0,
    seed=0,
    cpus=1,
    plot=True  # 新增绘图开关
):
    """
    Test a specific checkpoint on the test set
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
    log_file = os.path.join(log_path, f"test-{model_name}-{dataset}-{now}.log")
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

    # --------------------------- plot results --------------------------- #
    if plot:
        figure_path = os.path.join(project_root, "figure")
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        now_fig = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        num_samples_to_plot = min(5, y_true.shape[0])  # 可以调整绘图样本数

        for idx in range(num_samples_to_plot):
            plt.figure(figsize=(12, 6))
            plt.plot(y_true[idx, :, 0], label="True")
            plt.plot(y_pred[idx, :, 0], label="Predicted")
            plt.title(f"Sample {idx+1} Prediction vs True ({dataset})")
            plt.xlabel("Time step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            save_name = os.path.join(figure_path, f"{dataset}_sample{idx+1}_{now_fig}.png")
            plt.savefig(save_name)
            plt.close()
            print_log(f"Saved figure: {save_name}", log=log)

    # Also print to console
    print("\n" + "="*50)
    print(f"Test Results for {checkpoint_path}")
    print("="*50)
    print(out_str)
    print(f"Inference time: {end - start:.2f} s")
    print(f"Detailed log saved to: {log_file}")

    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True, 
                       help="Path to checkpoint file")
    parser.add_argument("-d", "--dataset", type=str, default="METRLA", 
                       help="Dataset name (default: METRLA)")
    parser.add_argument("-g", "--gpu_num", type=int, default=0, 
                       help="GPU number to use (default: 0)")
    parser.add_argument("--seed", type=int, default=0, 
                       help="Random seed (default: 0)")
    parser.add_argument("--cpus", type=int, default=1, 
                       help="Number of CPUs to use (default: 1)")
    parser.add_argument("--no-plot", action="store_true", 
                        help="Disable plotting of prediction results")
    
    args = parser.parse_args()

    test_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        gpu_num=args.gpu_num,
        seed=args.seed,
        cpus=args.cpus,
        plot=not args.no_plot
    )
