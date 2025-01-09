import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_scheduler, PreTrainedTokenizer, PreTrainedTokenizerFast
from spikingjelly.activation_based import functional
from typing import Union

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
        super().__init__(model, tokenizer, config, **kwargs)
        self.model.resize_token_embeddings(len(tokenizer))

        self.max_step = config.max_train_steps
        self.current_step = 0
        self.save_step_interval = config.save_step_interval

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
        self.fabric.launch()

        optimizer = self._get_optimizer()

        # 恢复权重
        if self.resume:
            model, optimizer, resume_step = self._load_checkpoint(self.model, optimizer)
            # 如果是从上一个 checkpoint 恢复， 则跳过已经训练过的步数
            if self.resume_key == self.config.name:
                self.current_step = resume_step

        model = self.fabric.setup(self.model)
        self.optimizer = self.fabric.setup_optimizers(optimizer)
        lr_scheduler = self._get_scheduler(optimizer)
        learner = Learner(model)

        train_dataloader = self._create_dataloader(
            train_dataset, self.config.batch_size
        )
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = self._create_dataloader(
                val_dataset, self.config.val_batch_size
            )
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        max_train_steps = self.config.max_train_steps
        pbar = tqdm(train_dataloader, total=max_train_steps, desc="Training")
        for batch in pbar:
            if self.current_step >= self.config.max_train_steps:
                break
            with self.fabric.no_backward_sync(model):
                loss_dict = learner(batch)
                loss = loss_dict["total_loss"]
                real_loss_ = loss_dict["real_loss"].mean()
                self.fabric.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            functional.reset_net(model)
            optimizer.zero_grad()
            self._auto_save_pretrained()

            self.log(
                LogFieldnames(
                    step=self.current_step,
                    train_loss=loss.item(),
                )
            )
            if loss.item():
                pbar.set_description(
                    f"Training: real_loss {real_loss_.item():.4f} | total_loss {loss.item():.4f}"
                )

            self.current_step += 1


BaseTrainer.register(SpikeBertForPreTraining.__name__, Trainer)
