import json
import torch
import torch.nn as nn
from typing import Optional, Union
from pathlib import Path
from .lib.inject import inject_script
from .utils import MODEL_PATH, model as model_utils


class Predicter:
    def __init__(
        self,
        name: str,
        model: nn.Module,
        device: str = "cuda",
        **kwargs,
    ):
        self.name = name
        self.model = model.to(device)
        self.device = device

    _subclasses = dict()

    @staticmethod
    def register(sub_class_name, sub_class):
        if sub_class_name in Predicter._subclasses:
            print(f"Name {sub_class_name} already exists in the Predicter registry.")
            return

        if not issubclass(sub_class, Predicter):
            raise ValueError(f"Subclass {sub_class} must be a subclass of Predicter.")

        Predicter._subclasses[sub_class_name] = sub_class

    def __new__(
        cls,
        name: str,
        model,
        device: str = "cuda",
        **kwargs,
    ):
        sub_class_name = model.__class__.__name__

        if sub_class_name in Predicter._subclasses:
            # 在 _subclasses 寻找注册的子类，如果存在则返回该子类的实例
            return super().__new__(Predicter._subclasses[sub_class_name])
        else:
            # 否则返回父类的实例
            return super().__new__(cls)

    @classmethod
    def get_root(cls) -> Path:
        return MODEL_PATH

    @classmethod
    def get_save_directory(cls, name: str) -> Path:
        save_directory = cls.get_root() / name
        return save_directory

    @classmethod
    def get_fork_directory(cls, fork: str) -> Optional[Path]:
        return model_utils.get_fork_directory(cls.get_root(), fork)

    @classmethod
    def get_checkpoint(cls, name: str, checkpoint: Optional[str] = None) -> Path:
        save_directory = cls.get_save_directory(name)
        return model_utils.get_checkpoint(save_directory, checkpoint)

    @classmethod
    def get_model_config_json(cls, name: str) -> dict:
        save_directory = cls.get_save_directory(name)
        return model_utils.get_model_config_json(save_directory)

    @classmethod
    def get_external_config_json(cls, name: str) -> Optional[dict]:
        save_directory = cls.get_save_directory(name)
        return model_utils.get_external_config_json(save_directory)

    @classmethod
    def get_trainer_config_json(cls, name: str) -> dict:
        save_directory = cls.get_save_directory(name)
        return model_utils.get_trainer_config_json(save_directory)

    @classmethod
    def get_model(cls, name: str, checkpoint: Optional[str] = None) -> nn.Module:
        """
        从保存的模型中加载模型。子类必须重写这个方法。
        """
        checkpoint_path = cls.get_checkpoint(name, checkpoint)
        pth_dict = torch.load(checkpoint_path, weights_only=True)
        sub_class_name = pth_dict.get("model_class")

        if sub_class_name is None:
            raise ValueError("model_class not found in config.json")
        sub_class = cls._subclasses.get(sub_class_name)
        if sub_class is None:
            raise ValueError(f"model_class {sub_class_name} not registered")

        print(f"Loading model {sub_class_name} from {checkpoint_path}")
        return sub_class.get_model(name, checkpoint)

    @classmethod
    def inject_script(cls, model, name: str):
        external_config = cls.get_external_config_json(name)
        if external_config:
            fork = external_config.get("fork", None)
            fork_directory = cls.get_fork_directory(fork)
            if fork_directory is not None:
                model = inject_script(model, fork_directory)
        return model

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
        return cls(name, model, device=device)

    def _predict(self, ctx):
        """
        单次执行预测的过程，一般子类需要重写这个方法
        """
        out = self.model(ctx)
        out = out.view(-1, out.size(-1))[-1:]
        return out.argmax(1)

    def predict(self, prompt: str, ctx_len: int = 1024):
        """
        预测的主要接口，输入一个 prompt，返回预测的结果
        """
        raise NotImplementedError("Method predict not implemented")
