########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import math
import torch
from tqdm.auto import tqdm
from itertools import cycle
from torch.utils.data.dataloader import DataLoader
from spikingjelly.activation_based import functional
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union

from .model import GPT as SpikeGPT
from .config import TrainerConfig
from ...main import Trainer as BaseTrainer, LogFieldnames
from darkit.core.utils.dataset import create_tokenized_gpt_dataset


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: SpikeGPT,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        config: TrainerConfig,
        **kwargs,
    ):
        super().__init__(model, tokenizer, config, **kwargs)
        # import wandb  # comment this if you don't have wandb
        # print('logging to wandb... (comment it if you don\'t have wandb)')
        # 启用 cuDNN 的自动优化功能，cuDNN 将会自动选择最优的卷积算法来加速计算。
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True  # 允许使用 TF32 精度进行计算。
        # 在矩阵乘法中允许使用 TF32 精度进行计算。
        torch.backends.cuda.matmul.allow_tf32 = True

        self.config = config
        self.avg_loss = -1
        self.min_dev_loss = 100
        self.dev_loss = -1
        self.lr = 0.0

        # 用于控制自动保存逻辑
        self.max_step = config.max_step
        self.current_step = 0
        self.save_step_interval = config.save_step_interval

    def _create_dataloader(self, dataset, batch_size):
        ctx_len = self.model.config.ctx_len

        tokenized_ds = create_tokenized_gpt_dataset(
            dataset, self.tokenizer, ctx_len=ctx_len + 1
        )
        return DataLoader(
            tokenized_ds,  # type: ignore
            shuffle=False,
            pin_memory=self.config.num_workers > 0,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
        )

    def _update_lr(self, optimizer, y):
        if self.config.lr_decay:  # 如果配置中启用了学习率衰减
            # number of tokens processed this step (i.e. label is not -100)
            self.tokens += (y >= 0).sum()
            lr_final_factor = self.config.lr_final / self.config.learning_rate
            if self.tokens < self.config.warmup_tokens:
                # linear warmup
                lr_mult = lr_final_factor + (1 - lr_final_factor) * float(
                    self.tokens
                ) / float(self.config.warmup_tokens)
            else:
                # cosine learning rate decay
                progress = float(self.tokens - self.config.warmup_tokens) / float(
                    max(1, self.config.final_tokens - self.config.warmup_tokens)
                )
                lr_mult = (0.5 + lr_final_factor / 2) + (
                    0.5 - lr_final_factor / 2
                ) * math.cos(
                    math.pi * progress
                )  # better 1.0 ~ 0.1
            lr = self.config.learning_rate * lr_mult
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            lr = self.config.learning_rate
        return lr

    def train(self, train_dataset, valid_dataset=None, is_train=True):
        self.fabric.launch()
        self.fabric.seed_everything(3407)

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = raw_model.configure_optimizers(self.config)

        # 如果指定了 resume， 则读取 checkpoint 的权重
        if self.resume:
            model, optimizer, resume_step = self._load_checkpoint(raw_model, optimizer)
            # 如果是从上一个 checkpoint 恢复， 则跳过已经训练过的步数
            print(f"resume_key: {self.resume_key}, config.name: {self.config.name}")
            if self.resume_key == self.config.name:
                print(f"resume_step: {resume_step}")
                self.current_step = resume_step

        model = self.fabric.setup(raw_model)
        self.optimizer = self.fabric.setup_optimizers(optimizer)

        loader = self._create_dataloader(train_dataset, self.config.batch_size)
        loader = self.fabric.setup_dataloaders(loader)

        pbar = tqdm(
            enumerate(cycle(loader)),
            total=(self.config.max_step - self.current_step),
            disable=not self._is_master_process(),
        )
        self.tokens = 0  # counter used for learning rate decay
        dev_loss_all = 0
        model.train(is_train)
        for it, train_data in pbar:
            if self.current_step >= self.config.max_step:
                break
            x = train_data[:, 0 : model.config.ctx_len].contiguous()
            y = train_data[:, 1 : model.config.ctx_len + 1].contiguous()

            with torch.set_grad_enabled(is_train):
                loss = model(x, y)  # forward the model
                functional.reset_net(model)
                self.fabric.backward(loss)

            if is_train:  # backprop and update the parameters
                if self.config.grad_norm_clip > 0:
                    self.fabric.clip_gradients(
                        model, optimizer, max_norm=self.config.grad_norm_clip
                    )

                optimizer.step()
                optimizer.zero_grad()

                self.lr = self._update_lr(optimizer, y)
                now_loss = loss.item()  # report progress

                # log training loss
                self.log(
                    LogFieldnames(
                        step=self.current_step * self.config.batch_size,
                        train_loss=now_loss,
                    )
                )

                if self.avg_loss < 0:
                    self.avg_loss = now_loss
                else:
                    factor = 1 / (it + 1)
                    self.avg_loss = self.avg_loss * (1.0 - factor) + now_loss * factor

                pbar.set_description(
                    f"step {self.current_step}: ppl {math.exp(self.avg_loss):.2f} loss {self.avg_loss:.4f} lr {self.lr:e}"
                )
                self._auto_save_pretrained()
                self.current_step += 1
            else:
                dev_loss_all += loss.item()

        if not is_train:
            self.dev_loss = dev_loss_all / len(loader)


BaseTrainer.register(SpikeGPT.__name__, Trainer)
