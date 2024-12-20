import json
import importlib
from pathlib import Path
from typing import Optional, Literal
from ..utils import MODEL_PATH


class ForkNetowrk:
    def __init__(
        self,
        name: str,
        model: str = "",
        m_conf: dict = {},
        t_conf: dict = {},
        base_directory: Path = MODEL_PATH,
    ):
        self.name = name
        self.model = model
        self.m_conf = {k: v for k, v in m_conf.items() if v != ""}
        self.t_conf = {k: v for k, v in t_conf.items() if v != ""}
        self.base_directory = base_directory

    @property
    def save_directory(self) -> Optional[Path]:
        if self.name is not None:
            save_directory = self.base_directory / self.name
            return save_directory
        else:
            return None

    @property
    def metadata_config_save_path(self) -> Optional[Path]:
        if self.save_directory:
            return self.save_directory / "metadata.json"
        else:
            return None

    def create_network(self):
        raise NotImplementedError

    def save(self):
        if self.save_directory is not None:
            if not self.save_directory.exists():
                self.save_directory.mkdir(parents=True)
            else:
                raise ValueError("save_directory already exists")
            with open(self.metadata_config_save_path, "w") as f:  # type: ignore
                external_conf = {
                    "model": self.model,
                    "m_conf": self.m_conf,
                    "t_conf": self.t_conf,
                }
                json.dump(external_conf, f)

        else:
            raise ValueError("save_directory is None")

    @classmethod
    def from_fork_name(cls, name: str, base_directory: Optional[Path] = None):
        if base_directory is None:
            fork = cls(name)
        else:
            fork = cls(name, base_directory=base_directory)
        if fork.metadata_config_save_path and fork.metadata_config_save_path.exists():
            with open(fork.metadata_config_save_path, "r") as f:
                metadata_conf = json.load(f)
                fork.model = metadata_conf["model"]
                fork.m_conf = metadata_conf["m_conf"]
                fork.t_conf = metadata_conf["t_conf"]
        else:
            print(f"metadata config not found for {name}")
        return fork
