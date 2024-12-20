import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .model import GPT, GPTConfig
from .model_run import RWKV_RNN, RWKVConfig
from ...main import Predicter as BasePredicter
from typing import Optional, Union


class SpikeGPTPredicter(BasePredicter):
    def __init__(
        self,
        name,
        model: RWKV_RNN,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(name, model, tokenizer=tokenizer, device=device, **kwargs)

    @classmethod
    def get_model(cls, name: str, checkpoint: Optional[str] = None):
        checkpoint_path = cls.get_checkpoint(name, checkpoint)
        config_dict = cls.get_model_config_json(name)
        config = GPTConfig(**config_dict)
        model = GPT(config)
        pth_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(pth_dict.get("model"))
        return model


BasePredicter.register(GPT.__name__, SpikeGPTPredicter)


class RWKVRNNPredicter(BasePredicter):
    def __init__(
        self,
        name: str,
        model: RWKV_RNN,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(name, model, tokenizer=tokenizer, device=device, **kwargs)

    @classmethod
    def get_model(cls, name: str, checkpoint: Optional[str] = None):
        checkpoint_path = cls.get_checkpoint(name, checkpoint)
        config_dict = cls.get_model_config_json(name)
        config = RWKVConfig(**config_dict)
        pth_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = RWKV_RNN(config, pth_dict.get("model"))
        return model

    def _predict(self, ctx, state=None, mem1=None, mem2=None, preprocess_only=False):
        out, state, mem1, mem2 = self.model(ctx, state, mem1, mem2, preprocess_only)
        ttt = out.argmax(-1)
        return ttt, state, mem1, mem2

    def predict(self, prompt: str, ctx_len: int = 1024):
        state, mem1, mem2 = None, None, None

        ctx = self.tokenizer.encode(prompt)

        for i in range(len(ctx), ctx_len):
            ttt, state, mem1, mem2 = self._predict(ctx, state, mem1, mem2)
            if ttt == self.tokenizer.eos_token_id:
                break
            char = self.tokenizer.decode(ttt)
            yield char


BasePredicter.register(RWKV_RNN.__name__, RWKVRNNPredicter)
