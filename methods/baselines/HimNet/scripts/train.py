import argparse
import numpy as np
import os
import torch
import datetime
import time
import matplotlib.pyplot as plt
import yaml
import json
import sys
import copy

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.utils import (
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from lib.losses import loss_select
from model.HimNet import HimNet
from model.HimNetRunner import HimNetRunner

# ! X shape: (B, T, N, C)


def train(
    model,
    runner,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    max_epochs=200,
    early_stop=10,
    compile_model=False,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    if torch.__version__ >= "2.0.0" and compile_model:
        model = torch.compile(model)
        print_log("Model compilation enabled", log=log)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    print_log("Starting training process...", log=log)
    print_log(f"Training set batches: {len(trainset_loader)}", log=log)
    print_log(f"Validation set batches: {len(valset_loader)}", log=log)
    print_log(log=log)

    for epoch in range(max_epochs):
        # Training phase
        print_log(f"Epoch {epoch+1}/{max_epochs} - Training...", log=log)
        train_loss = runner.train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion
        )
        train_loss_list.append(train_loss)

        # Validation phase
        print_log(f"Epoch {epoch+1}/{max_epochs} - Validating...", log=log)
        val_loss = runner.eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)

        # Print progress every epoch
        print_log(
            datetime.datetime.now(),
            "Epoch",
            epoch + 1,
            " \tTrain Loss = %.5f" % train_loss,
            "Val Loss = %.5f" % val_loss,
            log=log,
        )

        # Checkpoint for best model
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            print_log(f"New best model at epoch {epoch+1} with val_loss = {val_loss:.5f}", log=log)
            
            # Save checkpoint immediately when new best is found
            if save:
                checkpoint_path = save.replace('.pt', f'_epoch{epoch+1}.pt')
                torch.save(best_state_dict, checkpoint_path)
                print_log(f"Checkpoint saved: {checkpoint_path}", log=log)
        else:
            wait += 1
            if wait >= early_stop:
                print_log(f"Early stopping triggered at epoch {epoch+1}", log=log)
                break

    print_log("Training completed, loading best model...", log=log)
    model.load_state_dict(best_state_dict)

    print_log("Evaluating final model performance...", log=log)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(
        *runner.predict(model, trainset_loader)
    )
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*runner.predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
        print_log(f"Final model saved: {save}", log=log)
    return model


@torch.no_grad()
def test_model(model, runner, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

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


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="METRLA")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-c", "--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=1)
    args = parser.parse_args()

    if not args.seed:
        args.seed = np.random.randint(1, 10000)

    seed_everything(args.seed)
    set_cpu_num(args.cpus)

    # Set CUDA visible devices for multi-GPU training
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # Use all available GPUs
            gpu_ids = list(range(torch.cuda.device_count()))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            print(f"Using GPUs: {gpu_ids}")
        else:
            # Use single GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_num}"
            print(f"Using GPU: {args.gpu_num}")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()

    # Get baseline root directory (HimNet location)
    baseline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Get TrafficFM-main root directory (3 levels up: scripts -> HimNet -> baselines -> methods -> TrafficFM-main)
    trafficfm_root = os.path.dirname(os.path.dirname(os.path.dirname(baseline_root)))
    # Data path points to TrafficFM-main/data/processed/HimNet/
    data_path = os.path.join(trafficfm_root, "data", "processed", "HimNet", dataset)
    project_root = baseline_root  # Keep for model config and logs

    model_name = "HimNet"

    config_path = os.path.join(project_root, "model", f"{model_name}.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    # cfg.get(key, default_value=None): no need to write in the config if not used
    # cfg[key]: must be assigned in the config, else KeyError
    if cfg.get("pass_device"):
        cfg["model_args"]["device"] = DEVICE

    model = HimNet(**cfg["model_args"])
    # Enable multi-GPU training for 2 GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(DEVICE)

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(project_root, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
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

    # --------------------------- set model saving path -------------------------- #

    save_path = os.path.join(project_root, "saved_models")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = loss_select(cfg.get("loss", dataset))(**cfg.get("loss_args", {}))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 0.001),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.get("milestones", []),
        gamma=cfg.get("lr_decay_rate", 0.1),
    )

    # ----------------------------- set model runner ----------------------------- #

    runner = HimNetRunner(cfg, device=DEVICE, scaler=SCALER, log=log)

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(f"Seed = {args.seed}", log=log)
    print_log(f"Dataset: {dataset}", log=log)
    print_log(f"Input steps: {cfg.get('in_steps', 'N/A')}", log=log)
    print_log(f"Output steps: {cfg.get('out_steps', 'N/A')}", log=log)
    print_log(f"Device: {DEVICE}", log=log)
    print_log(f"Using {torch.cuda.device_count()} GPU(s)" if torch.cuda.is_available() else "Using CPU", log=log)
    
    # Simplified model info - only print basic info instead of full summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_log(f"Total parameters: {total_params:,}", log=log)
    print_log(f"Trainable parameters: {trainable_params:,}", log=log)
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss function: {criterion._get_name()}", log=log)
    print_log(f"Starting training with {cfg.get('max_epochs', 200)} max epochs", log=log)
    print_log(f"Early stopping patience: {cfg.get('early_stop', 10)}", log=log)
    print_log(log=log)

    model = train(
        model,
        runner,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        compile_model=args.compile,
        verbose=1,
        log=log,
        save=save,
    )

    test_model(model, runner, testset_loader, log=log)

    log.close()
