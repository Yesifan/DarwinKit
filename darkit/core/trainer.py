import os
import json
import inspect
import torch
import torch.nn as nn
from abc import abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional


from .lib.inject import inject_script
from .utils import MODEL_PATH, CSVLogger, get_local_ip, model as model_utils


@dataclass
class LogFieldnames:
    step: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None


@dataclass
class TrainerConfig:
    name: str = "cpu"
    model_type = "RWKV"
    device = None
    max_step = 100
    device_num = 1
    num_workers = 0  # for DataLoader

    # 模型评估设置
    eval_iters = 100
    eval_step_interval = 5

    # 模型保存设置
    save_step_interval = 10


class Trainer:
    """
    kwargs:
        - fork (str): 从分叉模型开始训练
        - resume (str): 从指定 checkpoint 读取权重继续训练, example: key:checkpoint
        - enable_server (bool): 是否启动可视化服务
    """

    _visual_series = {"train_loss": ["step", "train_loss"]}

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        **kwargs,
    ):

        self.model = model.to(config.device)
        self.device = config.device
        self.config = config
        self.fork = kwargs.get("fork", None)

        # 恢复模型设置
        self.resume = kwargs.get("resume", None)
        if self.resume:
            if ":" in self.resume:
                self.resume_key, self.resume_ckpt = self.resume.split(":")
            else:
                self.resume_key, self.resume_ckpt = self.resume, None

        # 模型评估设置
        self.eval_iters = getattr(self, "eval_iters", config.eval_iters)
        self.eval_step_interval = getattr(
            self, "eval_step_interval", config.eval_step_interval
        )

        # 模型保存设置
        self.max_step = getattr(self, "max_step", config.max_step)
        self.current_step = getattr(self, "current_step", 0)
        self.save_step_interval = getattr(
            self, "save_step_interval", config.save_step_interval
        )

        self._init_save_file()

        # 初始化 logger
        log_fieldnames_cls = kwargs.get("log_fieldnames", LogFieldnames)
        log_fieldnames = [field.name for field in fields(log_fieldnames_cls)]
        self._init_logger(log_fieldnames)

        # 加载分叉模型的修改代码
        self._inject_fork_script()

        self.__start_server(**kwargs)

    def _is_master_process(self):
        return True

    def __init_pid(self):
        # 将当前进程的 pid 写入文件
        pid = os.getpid()
        if self.save_directory:
            with open(self.save_directory / "pid", "w") as f:
                f.write(f"{pid}")

    def __del_pid(self):
        if self.save_directory:
            pid_file = self.save_directory / "pid"
            if pid_file.exists():
                os.remove(pid_file)

    def _init_save_file(self):
        if self.save_directory:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
                self._save_model_config()
                self._save_trainer_config()
                self._save_external_config()
                self._copy_model_code()
            else:
                raise ValueError(f"Model {self.save_directory} already exists")

    def _init_logger(self, log_fieldnames):
        if self.save_directory and log_fieldnames:
            filename = self.save_directory / "train_log.csv"
            self.csv_logger = CSVLogger(filename=filename, fieldnames=log_fieldnames)
            print(f"Logger initialized at {filename}")

    def _get_checkpoint_path(self):
        key_path = self.root / self.resume_key
        return model_utils.get_checkpoint(key_path, self.resume_ckpt)

    def _load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """将 checkpoint 加载到模型中"""
        if self.resume:
            print(f"Model loaded from {self.resume}")
            checkpoint_path = self._get_checkpoint_path()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            return model, optimizer, checkpoint["current_step"]
        return model, optimizer, self.current_step

    def log_exception(self, e):
        if self.save_directory:
            with open(self.save_directory / "exception.log", "w") as f:
                f.write(str(e))

    def __start_server(self, **kwargs):
        self._enabel_server = kwargs.get("enable_server", False)
        self.server_prot = kwargs.get("server_prot", 8000)
        if self._enabel_server:
            from .utils.server import start_uvicorn

            self._local_ip = get_local_ip() or "127.0.0.1"

            server_thread = start_uvicorn(port=self.server_prot)
            server_thread.daemon = True
            server_thread.start()

            print(f"Server started at http://{self._local_ip}:{self.server_prot}")

    def __enter__(self):
        self.__init_pid()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del_pid()

    _subclasses = dict()

    @staticmethod
    def register(sub_class_name, sub_class):
        if sub_class_name in Trainer._subclasses:
            print(f"Name {sub_class_name} already exists in the Trainer registry.")
            return

        if not issubclass(sub_class, Trainer):
            raise ValueError(f"Subclass {sub_class} must be a subclass of Trainer.")

        Trainer._subclasses[sub_class_name] = sub_class

    def __new__(
        cls,
        model,
        config: TrainerConfig,
        **kwargs,
    ):
        sub_class_name = model.__class__.__name__

        if sub_class_name in Trainer._subclasses:
            # 当直接在父类中调用时，在 _subclasses 寻找注册的子类，如果存在则返回该子类的实例
            return super().__new__(Trainer._subclasses[sub_class_name])
        else:
            return super().__new__(cls)

    @property
    def root(self) -> Path:
        return MODEL_PATH / "base"

    @property
    def save_directory(self) -> Optional[Path]:
        model_name = self.config.name
        if model_name is not None and self._is_master_process():
            save_directory = self.root / model_name
            return save_directory
        else:
            return None

    @property
    def fork_directory(self) -> Optional[Path]:
        if self.fork:
            return self.root / "fork" / self.fork
        else:
            return None

    @property
    def modle_config_save_path(self) -> Optional[Path]:
        if self.save_directory:
            return self.save_directory / "config.json"
        else:
            return None

    @property
    def trainer_config_save_path(self) -> Optional[Path]:
        if self.save_directory:
            return self.save_directory / "trainer_config.json"
        else:
            return None

    @property
    def external_config_save_path(self) -> Optional[Path]:
        if self.save_directory:
            return self.save_directory / "external_config.json"
        else:
            return None

    @property
    def tokenizer_save_path(self) -> Optional[Path]:
        if self.save_directory:
            return self.save_directory / "tokenizer"
        else:
            return None

    @property
    def model_code_archive_path(self) -> Optional[Path]:
        if self.save_directory:
            return self.save_directory / "model.py"
        else:
            return None

    @property
    def is_name_exist(self) -> bool:
        if self.save_directory:
            return os.path.exists(self.save_directory)
        else:
            return False

    def log(self, data):
        if hasattr(self, "csv_logger"):
            # if self.csv_logger is not None:
            self.csv_logger.log(data)

    def _save_model(self, checkpoint="complete"):
        """
        Save the model state, optimizer state, and current step.
        """
        if self.save_directory:
            current_step_idx = self.current_step + 1
            save_path = self.save_directory / f"{checkpoint}.pth"
            save_dict = {
                "model_class": self.model.__class__.__name__,
                "state_dict": self.model.state_dict(),
                "current_step": current_step_idx,
            }
            if hasattr(self, "optimizer"):
                save_dict["optimizer_state_dict"] = self.optimizer.state_dict()
            torch.save(save_dict, save_path)

    def _save_model_config(self):
        if self.modle_config_save_path and not self.modle_config_save_path.exists():
            with open(self.modle_config_save_path, "w") as f:
                json.dump(self.model.config.__dict__, f)

    def _save_trainer_config(self):
        if self.trainer_config_save_path and not self.trainer_config_save_path.exists():
            with open(self.trainer_config_save_path, "w") as f:
                print(self.config.__dict__)
                json.dump(self.config.__dict__, f)

    def _save_external_config(self):
        external_config = {
            "model_class": self.model.__class__.__name__,
            "series": self._visual_series,
            "fork": self.fork,
        }
        if (
            self.external_config_save_path
            and not self.external_config_save_path.exists()
        ):
            with open(self.external_config_save_path, "w") as f:
                json.dump(external_config, f)

    def _copy_model_code(self):
        try:
            if self.save_directory:
                model_source_code = inspect.getsource(self.model.__class__)
                with open(self.model_code_archive_path, "w", encoding="utf-8") as f:
                    f.write(model_source_code)
        except OSError as e:
            print("Save model code failed:", e)
        except TypeError as e:
            print("Cannot retrieve source code for built-in class:", e)

    def save_pretrained(self, check_poinent="complete"):
        """
        保存模型的配置、tokenizer、模型参数。
        """
        save_directory = self.save_directory
        if save_directory:
            if not os.path.exists(save_directory / check_poinent):
                self._save_model(check_poinent)
                print(f"Model saved at {save_directory}")
            else:
                raise ValueError(f"Model {save_directory} already exists")

    def _save_train_info(self):
        pass

    def _auto_save_pretrained(self):
        """
        根据 Train 的相关参数，控制训练时的自动保存逻辑。
        """
        current_step_idx = self.current_step + 1
        # 如果设置 save_step_interval 为 0，则不保存 checkpoint
        if self.save_step_interval > 0:
            # 当当前步数（current_step_idx）为 max_step 或者是 save_step_interval 的倍数时保存模型
            if (
                current_step_idx == self.max_step
                or current_step_idx % self.save_step_interval == 0
            ):
                check_poinent = f"iter-{current_step_idx}-ckpt"
                self.save_pretrained(check_poinent=check_poinent)
                if self.save_directory:
                    print(
                        f"Model saved epoch {current_step_idx}/{self.max_step} at {check_poinent}"
                    )

    def _auto_validate(self, val_dataloader):
        """
        根据 Train 的相关参数，控制训练时的自动验证逻辑。
        """
        current_step_idx = self.current_step + 1

        if (
            val_dataloader is not None
            and current_step_idx % self.eval_step_interval == 0
        ):
            self.validate(val_dataloader)

    def _inject_fork_script(self):
        if self.fork and self.fork_directory:
            self.model = inject_script(self.model, self.fork_directory)

    @abstractmethod
    def _get_optimizer(self):
        raise NotImplementedError("train method is not implemented.")

    @abstractmethod
    @torch.no_grad()
    def validate(self, val_dataloader) -> torch.Tensor:
        raise NotImplementedError("train method is not implemented.")

    @abstractmethod
    def train(self, train_dataset, val_dataloader=None, **kwargs):
        raise NotImplementedError("train method is not implemented.")


import lightning as L
from lightning.fabric.strategies.ddp import DDPStrategy


class FabricTrainer(Trainer):
    _visual_series = {"train_loss": ["step", "train_loss"]}

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        **kwargs,
    ):
        process_group_backend = kwargs.get("process_group_backend", "nccl")
        self.fabric = L.Fabric(
            devices=config.device_num,
            accelerator=config.device,
            strategy=DDPStrategy(process_group_backend=process_group_backend),
        )
        super().__init__(model, config, **kwargs)

    def _save_model(self, checkpoint="complete"):
        # 使用 fabric 保存模型
        if self.save_directory:
            save_path = self.save_directory / f"{checkpoint}.pth"
            current_step_idx = self.current_step + 1
            save_dict = {
                "model_class": self.model.__class__.__name__,
                "state_dict": self.model,
                "current_step": current_step_idx,
            }
            if hasattr(self, "optimizer"):
                save_dict["optimizer_state_dict"] = self.optimizer
            self.fabric.save(save_path, save_dict)
