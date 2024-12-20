import json
import torch
import torch.nn as nn
from typing import Optional, Union
from pathlib import Path
from .utils import MODEL_PATH


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
    def get_save_directory(cls, name: str) -> Path:
        save_directory = MODEL_PATH / name
        return save_directory

    @classmethod
    def get_checkpoint(cls, name: str, checkpoint: Optional[str] = None) -> Path:
        save_directory = cls.get_save_directory(name)

        # 寻找文件夹下的最新的 checkpoint 的 name
        if checkpoint:
            # check if the checkpoint exists
            if not (save_directory / f"{checkpoint}.pth").exists():
                raise FileNotFoundError(f"checkpoint {checkpoint} not found")
            return save_directory / f"{checkpoint}.pth"
        try:
            checkpoint_path = max(
                save_directory.glob("*.pth"), key=lambda x: x.stat().st_ctime
            )
            # 去掉后缀
            return checkpoint_path
        except ValueError:
            raise FileNotFoundError(f"checkpoint not found in {save_directory}")

    @classmethod
    def get_model_config_json(cls, name: str) -> dict:
        save_directory = cls.get_save_directory(name)
        with open(save_directory / "config.json", "r") as f:
            config_dict = json.load(f)
        return config_dict

    @classmethod
    def get_external_config_json(cls, name: str) -> Optional[dict]:
        save_directory = cls.get_save_directory(name)
        external_config_path = save_directory / "external_config.json"
        if external_config_path.exists():
            with open(external_config_path, "r") as f:
                config_dict = json.load(f)
            return config_dict
        else:
            return None

    @classmethod
    def get_trainer_config_json(cls, name: str) -> dict:
        save_directory = cls.get_save_directory(name)
        with open(save_directory / "trainer_config.json", "r") as f:
            config_dict = json.load(f)
        return config_dict

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
    def from_pretrained(
        cls,
        name: str,
        device: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ):
        trainer_config = cls.get_trainer_config_json(name)
        device = device if device else trainer_config.get("device", "cuda")
        model = cls.get_model(name, checkpoint).to(device)
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
