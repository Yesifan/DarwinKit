import torch
from .SpikeLM import SpikeLM, SpikeLMConfig, TrainerConfig as SpikeLMTrainConfig

if torch.cuda.is_available():
    from .SpikeGPT import SpikeGPT, SpikeGPTConfig, TrainerConfig as SpikeGPTTrainConfig

    Metadata = {
        "SpikeLM": {
            "cls": SpikeLM,
            "model": SpikeLMConfig,
            "trainer": SpikeLMTrainConfig,
        },
        "SpikeGPT": {
            "cls": SpikeGPT,
            "model": SpikeGPTConfig,
            "trainer": SpikeGPTTrainConfig,
        },
        "SpikingLlama": {},
    }
else:
    Metadata = {
        "SpikeLM": {
            "cls": SpikeLM,
            "model": SpikeLMConfig,
            "trainer": SpikeLMTrainConfig,
        },
    }


__all__ = [
    "SpikeLM",
    "SpikeLMConfig",
    "SpikeLMTrainConfig",
    "SpikeGPT",
    "SpikeGPTConfig",
    "SpikeGPTTrainConfig",
]
