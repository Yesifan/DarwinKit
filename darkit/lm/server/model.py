import traceback
import threading
import subprocess
from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException

from darkit.cli.src.generate import CLI_NAME, dict_to_cmd_args
from ..utils import MODEL_PATH

# 创建一个 APIRouter 实例
router = APIRouter()


class StartTrainBody(BaseModel):
    command: str


@router.post("/model/{model_type}/train")
def model_train(body: StartTrainBody):
    # if config.tconf["name"] and (MODEL_PATH / config.tconf["name"]).exists():
    #     raise HTTPException(status_code=400, detail="Model name already exists")
    try:
        process = subprocess.Popen(
            body.command,
            shell=True,
            text=True,
            bufsize=1,  # 使用行缓冲
            universal_newlines=True,  # 确保文本模式
        )

        return {"status": "success", "message": "Task start running"}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


class TrainData(BaseModel):
    type: str
    fork: str
    dataset: str
    tokenizer: str
    m_conf: dict
    t_conf: dict


@router.post("/v2/model/train/")
def model_train_2(body: TrainData):
    fork = body.fork

    name = body.t_conf.get("name")
    if name is None or name == "":
        raise HTTPException(status_code=400, detail="Model name is required")
    if (MODEL_PATH / name).exists():
        raise HTTPException(status_code=400, detail="Model name already exists")
    if task_lock.locked():
        raise HTTPException(status_code=400, detail="Task is already running")
    try:
        m_conf_str = dict_to_cmd_args(body.m_conf)
        t_conf_str = dict_to_cmd_args(body.t_conf)
        fork_command = f"--fork {fork}" if fork is not None and fork != "" else ""
        command = f"{CLI_NAME} lm train {fork_command} --tokenizer {body.tokenizer} --dataset {body.dataset} {body.type} {m_conf_str} {t_conf_str}"

        print("Start training with command: ")
        print(command)
        process = subprocess.Popen(
            command,
            shell=True,
            text=True,
            bufsize=1,  # 使用行缓冲
            universal_newlines=True,  # 确保文本模式
        )

        return {"status": "success", "message": "Task start running"}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


task_lock = threading.Lock()


class ModelPredictConfig(BaseModel):
    prompt: str
    ctx_len: int


@router.post("/model/{model_name}/predict")
def model_predict(model_name: str, data: ModelPredictConfig):
    from .. import models
    from ..main import Predicter

    print(f"Predicting with model: {model_name} {data}")
    if task_lock.locked():
        raise HTTPException(status_code=400, detail="Aleady a model is predicting")
    with task_lock:
        try:
            predicter = Predicter.from_pretrained(model_name)

            return StreamingResponse(
                content=predicter.predict(data.prompt, data.ctx_len),
                media_type="text/plain",
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
