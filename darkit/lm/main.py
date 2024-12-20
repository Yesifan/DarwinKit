import os
import torch
from pathlib import Path
from typing import Optional
from torch.nn.modules import Module
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer

from darkit.core import (
    Trainer as BaseTrainer,
    Predicter as BasePredicter,
    LogFieldnames,
)
from darkit.core.trainer import TrainerConfig
from darkit.core.lib.inject import inject_script
from .utils import MODEL_PATH, FORK_MODEL_PATH

from typing import Union

__all__ = ["Trainer", "Predicter", "LogFieldnames", "TrainerConfig"]


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: Module,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        config: TrainerConfig,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        super().__init__(model, config, **kwargs)
        self._inject_fork_script()

    def __init_save_file__(self):
        if self.save_directory:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
                self._save_tokenizer()
                self._save_model_config()
                self._save_trainer_config()
                self._save_external_config()
                self._copy_model_code()
            else:
                raise ValueError(f"Model {self.save_directory} already exists")

    def _inject_fork_script(self):
        if self.fork and self.fork_directory:
            self.model = inject_script(self.model, self.fork_directory)

    def _save_tokenizer(self):
        if self.tokenizer_save_path and not self.tokenizer_save_path.exists():
            self.tokenizer.save_pretrained(self.tokenizer_save_path)

    @property
    def save_directory(self) -> Optional[Path]:
        model_name = self.config.name
        if model_name is not None:
            save_directory = MODEL_PATH / model_name
            return save_directory
        else:
            return None

    @property
    def fork_directory(self) -> Optional[Path]:
        if self.fork:
            return FORK_MODEL_PATH / self.fork
        else:
            return None


class Predicter(BasePredicter):
    def __init__(
        self,
        name: str,
        model: Module,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        device: str = "cuda",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        super().__init__(name, model, device, **kwargs)

    @classmethod
    def get_tokenizer(
        cls, name: str
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        save_directory = cls.get_save_directory(name)
        tokenizer = AutoTokenizer.from_pretrained(save_directory / "tokenizer")
        tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token else "<pad>"
        return tokenizer

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        device: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ):
        trainer_config = cls.get_trainer_config_json(name)
        external_config = cls.get_external_config_json(name)
        device = device if device else trainer_config.get("device", "cuda")
        model = cls.get_model(name, checkpoint).to(device)
        if external_config:
            fork = external_config.get("fork", None)
            fork_directory = cls.get_fork_directory(fork)
            if fork_directory is not None:
                model = inject_script(model, fork_directory)
        tokenizer = cls.get_tokenizer(name)
        return cls(name, model, tokenizer=tokenizer, device=device)  # type: ignore

    @classmethod
    def get_save_directory(cls, name: str) -> Path:
        save_directory = MODEL_PATH / name
        return save_directory

    @classmethod
    def get_fork_directory(cls, fork: str) -> Optional[Path]:
        if fork:
            return FORK_MODEL_PATH / fork
        else:
            return None

    def predict(self, prompt: str, ctx_len: int = 1024):
        """
        预测的主要接口，输入一个 prompt，返回预测的结果
        一般子类不需要重写这个方法
        """
        ctx = self.tokenizer.encode(prompt, return_tensors="pt")
        ctx = ctx.to(self.device)  # type: ignore

        for i in range(len(ctx), ctx_len):
            out = self._predict(ctx)
            if out[0] == self.tokenizer.eos_token_id:
                break
            char = self.tokenizer.decode(out)
            ctx = torch.cat((ctx, out.unsqueeze(0)), dim=1)
            yield char

        # prompt_len = len(prompt.split(' '))
        # fake_prompt = 'Towards energy-efficient artificial intelligence similar to the human brain, the bio-inspired spiking neural networks (SNNs) have emerged as a promising alternative to traditional deep learning models. Unlike conventional artificial neural networks, which rely on continuous activations, SNNs mimic the discrete and event-driven firing of neurons in the brain, enabling more efficient computation.'
        # print(f'[info] predict: {prompt}')
        # ctx = self.tokenizer.encode(prompt, return_tensors="pt")
        # ctx = ctx.to(self.device)  # type: ignore

        # fake_output = fake_prompt.split(' ')
        # import time
        # for i in range(prompt_len, ctx_len):
        #     time.sleep(0.1)
        #     yield fake_output[i] + ' '
