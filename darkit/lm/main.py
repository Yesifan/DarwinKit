import os
import torch
from pathlib import Path
from typing import Optional
from torch.nn.modules import Module
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer

from darkit.core import (
    FabricTrainer,
    Predicter as BasePredicter,
    LogFieldnames,
)
from darkit.core.trainer import TrainerConfig
from .utils import MODEL_PATH

from typing import Union

__all__ = ["Trainer", "Predicter", "LogFieldnames", "TrainerConfig"]


class Trainer(FabricTrainer):
    def __init__(
        self,
        model: Module,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        config: TrainerConfig,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        super().__init__(model, config, **kwargs)

    def _is_master_process(self):
        return self.fabric.is_global_zero

    def _init_save_file(self):
        if self.save_directory:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
                self._save_tokenizer()
                self._save_model_config()
                self._save_trainer_config()
                self._save_external_config()
                self._copy_model_code()
            elif self.resume and self.resume_key == self.config.name:
                print("Resuming training...")
            else:
                raise ValueError(f"Model {self.save_directory} already exists")

    def _save_tokenizer(self):
        if self.tokenizer_save_path and not self.tokenizer_save_path.exists():
            self.tokenizer.save_pretrained(self.tokenizer_save_path)

    @property
    def root(self) -> Path:
        return MODEL_PATH


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
    def get_root(cls) -> Path:
        return MODEL_PATH

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
        device = device if device else trainer_config.get("device", "cuda")
        model = cls.get_model(name, checkpoint).to(device)
        model = cls.inject_script(model, name)
        tokenizer = cls.get_tokenizer(name)

        return cls(name, model, tokenizer=tokenizer, device=device)  # type: ignore

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
