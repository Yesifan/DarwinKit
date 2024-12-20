import os
import json
import inspect
import torch
import torch.nn as nn
from abc import abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

from .utils import MODEL_PATH, CSVLogger, get_local_ip


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
    num_device = 1
    num_workers = 0  # for DataLoader

    # 模型评估设置
    eval_iters = 100
    eval_step_interval = 5

    # 模型保存设置
    save_step_interval = 10


class Trainer:
    """
    kwargs:
        - enable_server: 是否启动可视化服务
    """

    _visual_series = {"train_loss": ["step", "train_loss"]}

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        **kwargs,
    ):

        self.model = model
        self.device = config.device
        self.config = config
        self.fork = kwargs.get("fork", None)

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

        self.__init_save_file__()

        log_fieldnames_cls = kwargs.get("log_fieldnames", LogFieldnames)
        log_fieldnames = [field.name for field in fields(log_fieldnames_cls)]
        self.__init_logger__(log_fieldnames)

        self.__start_server__(**kwargs)

    def _is_master_process(self):
        return True

    def __init_pid__(self):
        # 将当前进程的 pid 写入文件
        pid = os.getpid()
        if self.save_directory:
            with open(self.save_directory / "pid", "w") as f:
                f.write(f"{pid}")

    def __del_pid__(self):
        if self.save_directory:
            pid_file = self.save_directory / "pid"
            if pid_file.exists():
                os.remove(pid_file)

    def __init_save_file__(self):
        if self.save_directory:
            if not os.path.exists(self.save_directory):
                os.makedirs(self.save_directory)
                self._save_model_config()
                self._save_trainer_config()
                self._save_external_config()
                self._copy_model_code()
            else:
                raise ValueError(f"Model {self.save_directory} already exists")

    def __init_logger__(self, log_fieldnames):
        if self.save_directory and log_fieldnames:
            filename = self.save_directory / "train_log.csv"
            self.csv_logger = CSVLogger(filename=filename, fieldnames=log_fieldnames)
            print(f"Logger initialized at {filename}")

    def log_exception(self, e):
        if self.save_directory:
            with open(self.save_directory / "exception.log", "w") as f:
                f.write(str(e))

    def __start_server__(self, **kwargs):
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
        self.__init_pid__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del_pid__()

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
    def save_directory(self) -> Optional[Path]:
        model_name = self.config.name
        if model_name is not None and self._is_master_process():
            save_directory = MODEL_PATH / model_name
            return save_directory
        else:
            return None

    @property
    def fork_directory(self) -> Optional[Path]:
        if self.fork:
            return MODEL_PATH / self.fork
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
    def model_copy_path(self) -> Optional[Path]:
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

    def _save_model(self, check_poinent="complete"):
        """
        model save {model, model_class, current_step}
        """
        if self.save_directory:
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "model_class": self.model.__class__.__name__,
                    "current_step": self.current_step,
                },
                self.save_directory / f"{check_poinent}.pth",
            )

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
                model_py_path = inspect.getfile(self.model.__class__)
                with open(model_py_path, "r", encoding="utf-8") as f:
                    model_source_code = f.read()
                    with open(
                        self.save_directory / "model.py", "w", encoding="utf-8"
                    ) as f:
                        f.write(model_source_code)
        except OSError as e:
            print("Save model code failed:", e)

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
        max_step, current_step, save_step_interval = (
            self.max_step,
            self.current_step,
            self.save_step_interval,
        )

        current_step_idx = current_step + 1
        if self.save_step_interval > 0:
            if (
                current_step_idx == max_step
                or current_step_idx % save_step_interval == 0
            ):
                check_poinent = f"iter-{current_step_idx}-ckpt"
                self.save_pretrained(check_poinent=check_poinent)
                if self.save_directory:
                    print(
                        f"Model saved epoch {current_step_idx}/{max_step} at {check_poinent}"
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

    @abstractmethod
    @torch.no_grad()
    def validate(self, val_dataloader) -> torch.Tensor:
        raise NotImplementedError("train method is not implemented.")

    @abstractmethod
    def train(self, train_dataset, val_dataloader=None, **kwargs):
        raise NotImplementedError("train method is not implemented.")
