import os
from pydantic import BaseModel
from fastapi import APIRouter

from ..utils import FORK_MODEL_PATH
from ..utils.fork import LMForkNetowrk


from typing import Optional

# 创建一个 APIRouter 实例
router = APIRouter()


class StartTrainBody(BaseModel):
    command: str


@router.get("/models")
def get_all_fork_models(name: Optional[str] = None):
    if not FORK_MODEL_PATH.exists():
        FORK_MODEL_PATH.mkdir(parents=True)
        return []
    # 返回 MODEL_PATH 下的所有文件夹名，按创建时间倒序排序
    all_models = sorted(
        [model for model in FORK_MODEL_PATH.iterdir() if model.is_dir()],
        key=os.path.getctime,
        reverse=True,
    )
    # 如果有 name 参数，返回模糊查询的模型
    if name:
        all_models = [model for model in all_models if name in model.name]
    all_models = [LMForkNetowrk.from_fork_name(name=model.name) for model in all_models]

    return all_models


@router.get("/model/data/{fork}")
def get_model_metadata(fork: str):
    model = LMForkNetowrk.from_fork_name(name=fork)
    return {
        "model": model.model,
        "m_conf": model.m_conf,
        "t_conf": model.t_conf,
    }
