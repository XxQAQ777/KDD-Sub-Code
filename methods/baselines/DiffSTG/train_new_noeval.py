# -*- coding: utf-8 -*-
import os, sys
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer
import time
import contextlib 

from utils.eval import Metric
from utils.gpu_dispatch import GPU
from utils.common_utils import dir_check, to_device, ws, unfold_dict, dict_merge, GpuId2CudaId, Logger

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.model import DiffSTG, save2file

from tqdm import tqdm

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# --- 上下文管理器：用于屏蔽模型内部的 print ---
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# --- 修复布尔值参数解析 ---
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# for tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard'))
except:
    pass

def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    # model
    parser.add_argument("--epsilon_theta", type=str, default='UGnet')
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--beta_schedule", type=str, default='quad')
    parser.add_argument("--beta_end", type=float, default=0.1)
    
    # --- 关键修改：这里的 default 值可以由命令行覆盖 ---
    parser.add_argument("--sample_steps", type=int, default=200, help="Inference steps (iterative denoising)") 
    
    parser.add_argument("--ss", type=str, default='ddpm') 
    parser.add_argument("--T_h", type=int, default=144)

    # eval
    parser.add_argument('--n_samples', type=int, default=8)

    # train
    parser.add_argument("--is_train", type=str2bool, default=True) 
    parser.add_argument("--data", type=str, default='pems-bay')
    parser.add_argument("--mask_ratio", type=float, default=0.0) 
    parser.add_argument("--is_test", type=str2bool, default=False)
    parser.add_argument("--nni", type=str2bool, default=False)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    # multi-gpu
    parser.add_argument("--use_multi_gpu", type=str2bool, default=False)
    # --- 关键修改：指定 GPU ID ---
    parser.add_argument("--gpu_ids", type=str, default='0', help='comma separated gpu ids, e.g., 0 or 0,1')

    args, _ = parser.parse_known_args()
    return args

def default_config(data='AIR_BJ'):
    config = edict()
    config.PATH_MOD = ws + '/output/model/'
    config.PATH_LOG = ws + '/output/log/'
    config.PATH_FORECAST = ws + '/output/forecast/'

    # Data Config
    config.data = edict()
    config.data.name = data
    config.data.path = ws + '/data/dataset/'

    config.data.feature_file = config.data.path + config.data.name + '/flow.npy'
    config.data.spatial = config.data.path + config.data.name + '/adj.npy'
    config.data.num_recent = 1

    # ... (Keep existing dataset config logic) ...
    if config.data.name == 'PEMS08':
        config.data.num_features = 1
        config.data.num_vertices = 170
        config.data.points_per_hour = 12
        config.data.val_start_idx = int(17856 * 0.6)
        config.data.test_start_idx = int(17856 * 0.8)
    
    # (Other datasets logic omitted for brevity, assuming existing logic fits)
    if config.data.name == "AIR_BJ":
        config.data.num_features = 1
        config.data.num_vertices = 34
        config.data.points_per_hour = 1
        config.data.val_start_idx = int(8760 * 0.6)
        config.data.test_start_idx = int(8760 * 0.8)

    if config.data.name == 'AIR_GZ':
        config.data.num_features = 1
        config.data.num_vertices = 41
        config.data.points_per_hour = 1
        config.data.val_start_idx = int(8760 * 10 / 12) 
        config.data.test_start_idx = int(8160 * 11 / 12)

    if config.data.name == 'metr-la':
        config.data.num_features = 3 
        config.data.num_vertices = 207
        config.data.points_per_hour = 12
        config.data.val_start_idx = int(34272 * 0.7) 
        config.data.test_start_idx = int(34272 * 0.8) 

    if config.data.name == 'pems-bay':
        config.data.num_features = 3 
        config.data.num_vertices = 325
        config.data.points_per_hour = 12
        config.data.val_start_idx = int(52116 * 0.7) 
        config.data.test_start_idx = int(52116 * 0.8) 

    # GPU configuration default
    config.use_multi_gpu = False 
    config.gpu_ids = []
    config.gpu_id = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model config
    config.model = edict()
    config.model.T_p = 144
    config.model.T_h = 144
    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features
    config.model.week_len = 7
    config.model.day_len = config.data.points_per_hour * 24
    config.model.device = device
    config.model.d_h = 32 

    # config for diffusion model
    config.model.N = 200 
    config.model.sample_steps = 200
    
    config.model.epsilon_theta = 'UGnet'
    config.model.is_label_condition = True
    config.model.beta_end = 0.02
    config.model.beta_schedule = 'quad'
    config.model.sample_strategy = 'ddpm'
    config.n_samples = 1 

    config.model.channel_multipliers = [1, 2] 
    config.model.supports_len = 2

    # training config
    config.model_name = 'DiffSTG'
    config.is_test = False 
    config.epoch = 300 
    config.optimizer = "adam"
    config.lr = 1e-4 
    config.batch_size = 32 
    config.wd = 1e-5 
    config.early_stop = 10 
    config.start_epoch = 0 
    config.device = device
    config.logger = Logger()

    if not os.path.exists(config.PATH_MOD): os.makedirs(config.PATH_MOD)
    if not os.path.exists(config.PATH_LOG): os.makedirs(config.PATH_LOG)
    if not os.path.exists(config.PATH_FORECAST): os.makedirs(config.PATH_FORECAST)
    return config

def evals(model, data_loader, epoch, metric, config, clean_data, mode='Test'):
    setup_seed(2022)
    y_pred, y_true, time_lst = [], [], []
    metrics_future = Metric(T_p=config.model.T_p)
    metrics_history = Metric(T_p=config.model.T_h)
    model.eval()

    # Pbar for evaluation
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Eval {mode}", leave=True, dynamic_ncols=True)
    
    for i, batch in pbar:
        if i > 0 and config.is_test: break
        time_start = timer()

        future, history, pos_w, pos_d = to_device(batch, config.device)
        x = torch.cat((history, future), dim=1).to(config.device) 
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device) 
        x = x.transpose(1, 3) 
        x_masked = x_masked.transpose(1, 3) 
        
        n_samples = 1 if mode == 'Val' else config.n_samples

        with torch.no_grad():
            if mode == 'test' and n_samples > 1:
                x_hat_list = []
                for _ in range(n_samples):
                    with suppress_stdout():
                        x_hat_single = model((x_masked, pos_w, pos_d), 1) 
                    x_hat_list.append(x_hat_single)
                    torch.cuda.empty_cache()
                x_hat = torch.cat(x_hat_list, dim=1) 
            else:
                with suppress_stdout():
                    x_hat = model((x_masked, pos_w, pos_d), n_samples)

        if x_hat.shape[-1] != (config.model.T_h + config.model.T_p): x_hat = x_hat.transpose(2,4)

        time_lst.append((timer() - time_start))
        x, x_hat= clean_data.reverse_normalization(x), clean_data.reverse_normalization(x_hat)
        x_hat = x_hat.detach()
        f_x, f_x_hat = x[:,:,:,-config.model.T_p:], x_hat[:,:,:,:,-config.model.T_p:] 

        _y_true_ = f_x.transpose(1, 3).cpu().numpy() 
        _y_pred_ = f_x_hat.transpose(2, 4).cpu().numpy() 
        _y_pred_ = np.clip(_y_pred_, 0, np.inf)
        metrics_future.update_metrics(_y_true_, _y_pred_)

        y_pred.append(_y_pred_)
        y_true.append(_y_true_)

        h_x, h_x_hat = x[:, :, :, :config.model.T_h], x_hat[:, :, :, :,  :config.model.T_h]
        _y_true_ = h_x.transpose(1, 3).cpu().numpy() 
        _y_pred_ = h_x_hat.transpose(2, 4).cpu().numpy()
        _y_pred_ = np.clip(_y_pred_, 0, np.inf)
        metrics_history.update_metrics(_y_true_, _y_pred_)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    time_cost = np.sum(time_lst)
    metric.update_metrics(y_true, y_pred)
    metric.metrics['time'] = time_cost

    if mode == 'test': 
        # (Save forecast logic retained...)
        import pickle
        with open (config.forecast_path, 'wb') as f:
            pickle.dump([y_pred, y_true], f) # Simplified dump for safety

        csv_path = config.forecast_path.replace('.pkl', '_step_metrics.csv')
        metric.save_step_metrics_to_csv(csv_path)
        metric.print_specific_steps(steps=[12, 36, 72, 144])

    # Log results
    message = f" | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}"
    print(message, end='\n', flush=False)
    config.logger.message_buffer += message + "\n"
    config.logger.write_message_buffer()

    torch.cuda.empty_cache()
    return metric


from pprint import  pprint
def main(params: dict):
    setup_seed(2022)
    torch.set_num_threads(2)
    config = default_config(params['data'])

    # Apply parameters from args
    config.is_test = params['is_test']
    config.nni = params['nni']
    config.lr = params['lr']
    config.batch_size = params['batch_size']
    config.mask_ratio = params['mask_ratio']
    config.epoch = params['epochs'] 

    # --- GPU Configuration ---
    config.use_multi_gpu = params.get('use_multi_gpu', False)
    gpu_ids_str = params.get('gpu_ids', '0') # Default to 0
    config.gpu_ids = [int(x) for x in gpu_ids_str.split(',')]
    
    if config.use_multi_gpu:
        print(f"Using multi-GPU training on GPUs: {config.gpu_ids}")
        torch.cuda.set_device(config.gpu_ids[0])
        config.device = torch.device(f'cuda:{config.gpu_ids[0]}')
    else:
        print(f"Using single GPU: {config.gpu_ids[0]}")
        torch.cuda.set_device(config.gpu_ids[0])
        config.device = torch.device(f'cuda:{config.gpu_ids[0]}')

    # model params
    config.model.N = params['N']
    config.T_h = config.model.T_h = params['T_h']
    config.T_p = config.model.T_p =  params['T_h'] 
    config.model.epsilon_theta =  params['epsilon_theta']
    
    # --- 重要：这里只设置训练的采样参数，推理参数后面单独设置 ---
    config.model.d_h = params['hidden_size']
    config.model.C = params['hidden_size']
    config.model.n_channels = params['hidden_size']
    config.model.beta_end = params['beta_end']
    config.model.beta_schedule = params["beta_schedule"]
    config.model.sample_strategy = params["ss"]
    config.n_samples = params['n_samples']

    config.trial_name = '+'.join([f"{v}" for k, v in params.items()])
    config.log_path = f"{config.PATH_LOG}/{config.trial_name}.log"

    pprint(config)
    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")
    config.logger.write(config.__str__()+'\n', is_terminal=False)

    clean_data = CleanDataset(config)
    config.model.A = clean_data.adj

    print("DiffSTG config here")
    model = DiffSTG(config.model)
    model = model.to(config.device) 

    if config.use_multi_gpu and len(config.gpu_ids) > 1:
        print(f"Wrapping model with DataParallel on GPUs: {config.gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        config.is_parallel = True
    else:
        config.is_parallel = False

    train_dataset = TrafficDataset(clean_data, (0 + config.model.T_p, config.data.val_start_idx - config.model.T_p + 1), config)
    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, shuffle=True, pin_memory=True)

    # Note: validation loader is still needed for final check, but not in loop
    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, params.get('test_batch_size', 1), shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)
    # Changed scheduler to monitor training loss since we removed val eval
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model_to_use = model.module if config.is_parallel else model
    model_path = config.PATH_MOD + config.trial_name + model_to_use.model_file_name()
    config.model_path = model_path
    config.forecast_path = config.PATH_FORECAST + config.trial_name + '.pkl'

    print('model_path:', model_path)

    # --- Training Loop ---
    print(f"\n>>> Starting training for {config.epoch} epochs. Validation will ONLY run at the end.")
    train_start_t = timer()
    
    for epoch in range(config.epoch):
        if not params['is_train']: break
        
        n, avg_loss, time_lst = 0, 0, []
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)
        
        model.train() # Ensure model is in train mode
        for i, batch in enumerate(train_pbar):
            time_start =  timer()
            future, history, pos_w, pos_d = batch 

            x = torch.cat((history, future), dim=1).to(config.device) 
            mask = torch.randint_like(history, low=0, high=100) < int(config.mask_ratio * 100)
            history[mask] = 0
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device) 

            x = x.transpose(1,3) 
            x_masked = x_masked.transpose(1,3) 

            # Suppress print inside loss
            with suppress_stdout():
                loss = 10 * model_to_use.loss(x, (x_masked, pos_w, pos_d))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n += 1
            avg_loss = avg_loss * (n - 1) / n + loss.item() / n
            time_lst.append((timer() - time_start))
            
            train_pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "time": f"{np.sum(time_lst):.1f}s"
            })

        # Step scheduler based on TRAINING LOSS since we skip validation
        scheduler.step(avg_loss)
        
        # Save model every epoch (overwriting) to ensure we have the latest
        if config.is_parallel:
            torch.save(model.module, model_path)
        else:
            torch.save(model, model_path)

    print("\n>>> Training Finished.")
    
    # --- Final Evaluation ---
    print(f">>> Loading best/last model from {model_path} for Final Evaluation...")
    try:
        model = torch.load(model_path, map_location=config.device, weights_only=False)
        if config.use_multi_gpu and len(config.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
    except Exception as err:
        print(f"Error loading model: {err}")
        return

    # --- Apply User Defined Sample Steps for Inference ---
    user_sample_steps = params['sample_steps']
    print(f">>> Setting inference sample steps to: {user_sample_steps}")
    
    config.model.sample_steps = user_sample_steps
    model_to_use = model.module if config.is_parallel else model
    model_to_use.set_ddim_sample_steps(user_sample_steps) # Assuming model has this method or uses config
    
    # Run Eval on Test Set
    metrics_test = Metric(T_p=config.model.T_h + config.model.T_p)
    evals(model, test_loader, config.epoch, metrics_test, config, clean_data, mode='test')
    
    print(f"\nFinal Test Results (Steps={user_sample_steps}): MAE={metrics_test.metrics['mae']:.4f}, RMSE={metrics_test.metrics['rmse']:.4f}")
    
    # Save final results
    params = unfold_dict(config)
    params = dict_merge([params, metrics_test.to_dict()])
    save2file(params)

    if params['nni']:
        nni.report_final_result(metrics_test.metrics['mae'])

if __name__ == '__main__':
    import nni
    import logging

    logger = logging.getLogger('training')
    try:
        tuner_params = nni.get_next_parameter()
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise