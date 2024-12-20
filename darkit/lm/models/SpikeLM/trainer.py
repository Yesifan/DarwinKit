import lightning as L
from lightning.fabric.strategies.ddp import DDPStrategy
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler, PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
from typing import Union
from spikingjelly.activation_based import functional

from darkit.core.utils.dataset import create_tokenized_bert_dataset
from darkit.core.utils.dataset.data_collator import MaskDataCollatorWithPadding

from .config import TrainerConfig
from .learner import Learner
from .spike_bert import SpikeBertForPreTraining
from ...main import Trainer as BaseTrainer, LogFieldnames


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: SpikeBertForPreTraining,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        config: TrainerConfig,
        **kwargs,
    ):
        process_group_backend = kwargs.get("process_group_backend", "nccl")
        self.fabric = L.Fabric(
            devices=config.device_num,  # 这个变量名和其他类是反过来的。注意一下
            accelerator=config.device,
            strategy=DDPStrategy(process_group_backend=process_group_backend),
        )
        super().__init__(model, tokenizer, config, **kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        model.resize_token_embeddings(len(tokenizer))

        self.max_step = config.max_train_steps
        self.current_step = 0
        self.save_step_interval = config.save_step_interval

    def _is_master_process(self):
        return self.fabric.is_global_zero

    def _save_model(self, checkpoint="complete"):
        """
        model save {model, model_class, current_step}
        """
        if self.save_directory:
            save_path = self.save_directory / f"{checkpoint}.pth"
            self.fabric.save(
                save_path,
                {
                    "model": self.model,
                    "model_class": self.model.__class__.__name__,
                    "current_step": self.current_step,
                    "optimizer": self.optimizer,
                },
            )

    def _create_dataloader(self, dataset, batch_size):
        max_seq_length = min(
            self.config.max_seq_length, self.tokenizer.model_max_length
        )
        data_collator = MaskDataCollatorWithPadding(
            tokenizer=self.tokenizer, padding="max_length", max_length=max_seq_length
        )
        tokenized_ds = create_tokenized_bert_dataset(dataset, self.tokenizer)
        return DataLoader(
            tokenized_ds,  # type: ignore
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size,
        )

    def _get_optimizer(self):
        weight_decay, learning_rate = (
            self.config.weight_decay,
            self.config.learning_rate,
        )
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)  # type: ignore

    def _get_scheduler(self, optimizer):
        tconf = self.config
        return get_scheduler(
            name=tconf.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=tconf.num_warmup_steps * tconf.gradient_accumulation_steps,
            num_training_steps=tconf.max_train_steps
            * tconf.gradient_accumulation_steps,
        )

    def train(self, train_dataset, val_dataset=None):
        fabric, tconfig = self.fabric, self.config
        max_train_steps = tconfig.max_train_steps

        fabric.launch()

        model = fabric.setup(self.model)
        learner = Learner(model)

        optimizer = self._get_optimizer()
        self.optimizer = fabric.setup_optimizers(optimizer)
        lr_scheduler = self._get_scheduler(optimizer)

        train_dataloader = self._create_dataloader(
            train_dataset, self.config.batch_size
        )
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = self._create_dataloader(
                val_dataset, self.config.val_batch_size
            )
            val_dataloader = fabric.setup_dataloaders(val_dataloader)

        pbar = tqdm(enumerate(train_dataloader), total=max_train_steps, desc="Training")
        for step, batch in pbar:
            if step >= max_train_steps:
                break
            with fabric.no_backward_sync(model):
                loss_dict = learner(batch)
                loss = loss_dict["total_loss"]
                real_loss_ = loss_dict["real_loss"].mean()
                fabric.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            functional.reset_net(model)
            optimizer.zero_grad()
            self._auto_save_pretrained()

            self.log(
                LogFieldnames(
                    step=step,
                    train_loss=loss.item(),
                )
            )
            if loss.item():
                pbar.set_description(
                    f"Training: real_loss {real_loss_.item():.4f} | total_loss {loss.item():.4f}"
                )

            self.current_step += 1


BaseTrainer.register(SpikeBertForPreTraining.__name__, Trainer)
