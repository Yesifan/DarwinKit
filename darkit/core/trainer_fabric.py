import torch
import torch.nn as nn
import lightning as L
from lightning.fabric.strategies.ddp import DDPStrategy

from .trainer import Trainer, TrainerConfig


class FabricTrainer(Trainer):
    _visual_series = {"train_loss": ["step", "train_loss"]}

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        **kwargs,
    ):
        process_group_backend = kwargs.get("process_group_backend", "nccl")
        self.fabric = L.Fabric(
            devices=config.device_num,
            accelerator=config.device,
            strategy=DDPStrategy(process_group_backend=process_group_backend),
        )
        super().__init__(model, config, **kwargs)

    def _save_model(self, checkpoint="complete"):
        # 使用 fabric 保存模型
        if self.save_directory:
            save_path = self.save_directory / f"{checkpoint}.pth"
            self.fabric.save(
                save_path,
                {
                    "model_class": self.model.__class__.__name__,
                    "state_dict": self.model,
                    "optimizer_state_dict": self.optimizer,
                    "current_step": self.current_step,
                },
            )
