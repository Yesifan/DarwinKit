import os
import time
import socket
import torch
import random
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from .csv_logger import CSVLogger

__all__ = ["CSVLogger"]


DARWIN_KIT_HOME = os.environ.get("DARWIN_KIT_HOME", "~/.cache/darwinkit")
HOME_PATH = os.path.expanduser(DARWIN_KIT_HOME)
MODEL_PATH = Path(HOME_PATH) / "models"
DATASET_PATH = Path(HOME_PATH) / "datasets"

TMP_PATH = Path(__file__).parent.parent.parent / "tmp"
PWA_PATH = Path(__file__).parent.parent / "web" / "build"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_local_ip():
    try:
        # 获取本地主机名
        hostname = socket.gethostname()
        # 通过主机名获取局域网 IP 地址
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except socket.error as e:
        print(f"Error getting local IP: {e}")
        return None


@contextmanager
def usetime(desc: str):
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{desc} 执行时间: {end_time - start_time} 秒")
