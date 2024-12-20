import numpy as np
from . import predicter, trainer
from .model import GPT as SpikeGPT
from .model_run import RWKV_RNN, RWKVConfig
from .config import GPTConfig as SpikeGPTConfig, TrainerConfig

# 由于 spikingjelly 的 0.0.0.0.14 版本中 activation_based/auto_cuda/base.py 使用了 np.int
# 而在 numpy 1.21.0 版本中，np.int 已经被弃用，因此这里将 np.int 替换为 np.int_
np.int = np.int_  # type: ignore

__all__ = (
    "TrainerConfig",
    "SpikeGPT",
    "SpikeGPTConfig",
    "RWKV_RNN",
    "RWKVConfig",
)
