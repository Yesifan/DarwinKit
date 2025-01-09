import json
from dataclasses import dataclass
from typing import Literal

from darkit.core import TrainerConfig as BaseTrainerConfig


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model.

    Args:
        vocab_size (int): 词汇表大小。表示可以处理不同token的数量。词汇表大小直接影响模型的参数数量，较大的词汇表意味着嵌入层和输出层的参数会更多。
        ctx_len (int): 上下文长度。模型在处理输入序列时所能考虑的最大序列长度。较长的上下文长度需要更多的计算资源。**Increase T_MAX in model.py if your ctx_len > 1024**
        model_type (str): 指定架构类型，有RWKV和RWKV-ffnPre两种供选择。'RWKV' (better for char-level English) or 'RWKV-ffnPre' (better in some cases)
        n_layer (int): Block层数。Block架构见论文Figure1或model.py的Block类。
        n_embd (int): 嵌入维度。决定了模型中嵌入层(embedding layer)的向量维度。
        **kwargs: Additional keyword arguments for custom configuration.

    Methods:
        save_pretrained(model_name): Save the configuration to a file.
        from_pretrained(model_name): Load the configuration from a file.

    """

    vocab_size: int = 30500
    ctx_len: int = 1024
    model_type: Literal["RWKV", "RWKV-ffnPre"] = "RWKV"
    n_layer: int = 12
    n_embd: int = 768


@dataclass
class RWKVConfig:
    vocab_size: int = 30500
    ctx_len: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    head_qk: int = 0
    pre_ffn: int = 0
    grad_cp: int = 0
    my_pos_emb: int = 0

    def __init__(self, vocab_size, ctx_len, n_layer, n_embd, **kwargs):
        import os
        import torch

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.RUN_DEVICE = "cuda"  # 'cuda' // 'cpu' (already fast)
        self.FLOAT_MODE = "fp32"  # fp16 (good for GPU, does not work for CPU) // fp32 (good for CPU) // bf16 (less accurate, but works for CPU)

        os.environ["RWKV_JIT_ON"] = (
            "1"  # '1' or '0'. very useful for GPU/CPU fp32, but might be harmful for GPU fp16. please benchmark !!!
        )
        os.environ["RWKV_RUN_DEVICE"] = self.RUN_DEVICE

        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.head_qk = kwargs.get("head_qk", 0)
        self.pre_ffn = kwargs.get("pre_ffn", 0)
        self.grad_cp = kwargs.get("grad_cp", 0)
        self.my_pos_emb = kwargs.get("my_pos_emb", 0)


@dataclass
class TrainerConfig(BaseTrainerConfig):
    device: str = "cuda"
    name: str = "SpikeGPT"
    model_type: str = "RWKV"
    batch_size: int = 2
    max_step: int = 100
    learning_rate: float = 4e-4
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    grad_norm_clip: float = 1.0
    lr_decay: bool = True  # linear warmup followed by cosine decay
    warmup_tokens: int = 0
    final_tokens: int = 0
    device_num: int = 1  # number of devices. eg. 1 for single GPU, 2 for two GPUs
    num_workers: int = 0  # for DataLoader
    lr_final: float = 1e-5
    save_step_interval: int = 100

    @property
    def betas(self):
        return (self.beta1, self.beta2)


SPIKEGPT_CONFIG_COMMENT = {
    # GPTConfig related configurations
    "vocab_size": {
        "description": "Vocabulary size. This represents the number of different tokens the model can handle. The vocabulary size directly affects the number of parameters in the model, with a larger vocabulary meaning more parameters in the embedding and output layers.",
        "range": "Typically between a few thousand to several hundred thousand, e.g., 30,000 to 50,000",
    },
    "ctx_len": {
        "description": "Context length. This is the maximum sequence length that the model can consider when processing input sequences. Longer context lengths require more computational resources.",
        "range": "Typically between 128 and 2048, depending on hardware capabilities and task requirements",
    },
    "model_type": {
        "description": "Specifies the architecture type, with options being RWKV and RWKV-ffnPre. 'RWKV' (better for character-level English) or 'RWKV-ffnPre' (better in some cases).",
        "range": "'RWKV' or 'RWKV-ffnPre'",
    },
    "n_layer": {
        "description": "Number of layers in the model. The block architecture can be found in the paper's Figure 1 or in the Block class in model.py.",
        "range": "Typically between 6 and 24 layers, depending on the model complexity and computational resources",
    },
    "n_embd": {
        "description": "Embedding dimension. This determines the vector dimension of the embedding layer in the model.",
        "range": "Common values are 128, 256, 512, 768, 1024, etc., depending on the model scale",
    },
    # TrainerConfig related configurations
    "device": {
        "description": "Device used for training, e.g., 'cuda' for GPU or 'cpu' for CPU.",
        "range": "'cuda', 'cpu'",
    },
    "name": {
        "description": "Name of the trainer, used to identify different training instances.",
        "range": "Custom string",
    },
    "max_epochs": {
        "description": "Maximum number of training epochs.",
        "range": "Positive integer, typically from 1 to a few hundred",
    },
    "epoch_length_fixed": {
        "description": "Fixed number of iterations per epoch.",
        "range": "Positive integer, usually determined by the dataset size and batch size",
    },
    "batch_size": {
        "description": "Number of samples per batch.",
        "range": "Positive integer, typically between 8 and 256",
    },
    "learning_rate": {
        "description": "Learning rate, which controls the speed of weight updates.",
        "range": "Typically between 1e-5 and 1e-2",
    },
    "beta1": {
        "description": "First momentum parameter for the Adam optimizer.",
        "range": "Typically between 0.8 and 0.99",
    },
    "beta2": {
        "description": "Second momentum parameter for the Adam optimizer.",
        "range": "Typically between 0.99 and 0.999",
    },
    "eps": {
        "description": "Numerical stability term for the Adam optimizer.",
        "range": "Typically very small, such as 1e-8",
    },
    "grad_norm_clip": {
        "description": "Gradient clipping threshold to prevent exploding gradients.",
        "range": "Typically between 0.5 and 5",
    },
    "lr_decay": {
        "description": "Whether to enable learning rate decay, with linear warmup followed by cosine decay.",
        "range": "Boolean, True or False",
    },
    "warmup_tokens": {
        "description": "Number of tokens processed during the learning rate warmup phase.",
        "range": "Non-negative integer, typically 0 or a few thousand to several million",
    },
    "final_tokens": {
        "description": "Total number of tokens processed, used for learning rate decay calculation.",
        "range": "Non-negative integer, typically 0 or tens of millions to billions",
    },
    "device_num": {
        "description": "Number of devices used, e.g., 1 for a single GPU, 2 for two GPUs.",
        "range": "Positive integer, depending on available hardware",
    },
    "num_workers": {
        "description": "Number of workers for the DataLoader, used for data loading.",
        "range": "Non-negative integer, typically 0 to the number of CPU cores",
    },
    "lr_final": {
        "description": "Final learning rate after decay.",
        "range": "Typically smaller than the initial learning rate, e.g., 1e-5",
    },
    "epoch_save_frequency": {
        "description": "How often (in epochs) to save the model. A value of 0 means no automatic saving.",
        "range": "Non-negative integer",
    },
}

META_INFO = {
    "SpikeGPT": {
        "model": GPTConfig,
        "trainer": TrainerConfig,
        "comment": SPIKEGPT_CONFIG_COMMENT,
    }
}
