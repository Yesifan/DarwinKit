import sys
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
import importlib.util
from pathlib import Path


class SpikingModule(nn.Module):
    def __init__(self, T, module):
        super().__init__()
        self.T = T
        self.module = module
        functional.set_step_mode(self, step_mode="m")

    def forward(self, x):
        # print("x shape", x.shape)
        x_seq = x.unsqueeze(0).expand(10, *x.shape)
        # print("x_seq shape", x_seq.shape)

        x_seq = self.module(x_seq)
        fr = x_seq.sum(0)
        # print("fr shape", fr.shape)

        return fr


def load_model_from_file(name: str, file_path: Path) -> torch.nn.Module:
    """
    动态加载上传的模型文件，假设模型类名称为 `MyModel`
    """
    module_name = file_path.stem  # 模块名为文件名
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, name):
        raise ValueError(f"模型文件中未找到 `{name}` 类")

    model_class = getattr(module, name)
    return model_class()
