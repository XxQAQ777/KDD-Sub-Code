import time
import csv
import os
import torch

from basicts.runners import BaseTimeSeriesForecastingRunner
from basicts.metrics import masked_mae, masked_rmse, masked_mape, masked_crps, masked_wasserstein


class STDMAERunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.metrics = cfg.get("METRICS", {
            "MAE": masked_mae,
            "RMSE": masked_rmse,
            "MAPE": masked_mape,
            "CRPS": masked_crps,
            "Wasserstein": masked_wasserstein
        })
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

        # Initialize epoch tracking
        self.epoch_start_time = None

    def init_training(self, cfg: dict):
        """Initialize training and log model parameters."""
        super().init_training(cfg)

        # Register additional meters for monitoring
        self.register_epoch_meter("train_loss", "train", "{:.4f}")
        self.register_epoch_meter("epoch_time", "train", "{:.2f}s", plt=False)
        self.register_epoch_meter("gpu_memory_mb", "train", "{:.2f}", plt=False)

        # Calculate and log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info("="*50)
        self.logger.info(f"Model Parameters Summary:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        self.logger.info("="*50)

    def on_epoch_start(self, epoch: int):
        """Record epoch start time and reset GPU memory stats."""
        super().on_epoch_start(epoch)
        self.epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(self, epoch: int):
        """Record epoch statistics."""
        # Calculate epoch time
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.update_epoch_meter("epoch_time", epoch_time)

        # Record GPU memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            self.update_epoch_meter("gpu_memory_mb", gpu_memory)

        super().on_epoch_end(epoch)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C].

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        future_data, history_data, long_history_data = data
        history_data        = self.to_running_device(history_data)      # B, L, N, C
        long_history_data   = self.to_running_device(long_history_data) # B, L, N, C
        future_data         = self.to_running_device(future_data)       # B, L, N, C

        history_data = self.select_input_features(history_data)
        long_history_data = self.select_input_features(long_history_data)

        # feed forward
        prediction= self.model(history_data=history_data, long_history_data=long_history_data, future_data=None, batch_seen=iter_num, epoch=epoch)

        batch_size, length, num_nodes, _ = future_data.shape
        assert list(prediction.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"

        # post process
        prediction = self.select_target_features(prediction)
        real_value = self.select_target_features(future_data)
        return prediction, real_value
