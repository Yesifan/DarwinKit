import torch
from dataclasses import fields
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .spike_bert import SpikeBertForPreTraining
from .config import SpikeLMConfig
from ...main import Predicter as BasePredicter

from typing import Optional, Union


class SpikeLMPredicter(BasePredicter):
    def __init__(
        self,
        name: str,
        model: SpikeBertForPreTraining,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        device: str = "cuda",
    ):
        super().__init__(name, model, tokenizer, device)

    @classmethod
    def get_model_config(cls, name: str):
        config_dict = cls.get_model_config_json(name)
        valid_keys = {f.name for f in fields(SpikeLMConfig)}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        config = SpikeLMConfig(**filtered_config)
        return config

    @classmethod
    def get_model(cls, name: str, checkpoint: Optional[str] = None):
        checkpoint_path = cls.get_checkpoint(name, checkpoint)
        config = cls.get_model_config(name)
        model = SpikeBertForPreTraining(config)
        checkpoint_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint_dict["model"], strict=True)
        return model

    def _predict(self, ctx):
        out = self.model(ctx.contiguous()).prediction_logits
        out = out.view(-1, out.size(-1))[-1:]
        return out.argmax(1)


BasePredicter.register(SpikeBertForPreTraining.__name__, SpikeLMPredicter)
