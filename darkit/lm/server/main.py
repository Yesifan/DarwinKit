import os
import csv
import json
import signal
import tarfile
import shutil
import tempfile
import asyncio
import aiofiles
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from fastapi.websockets import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response

from darkit.core.lib.options import get_models_options
from .model import router
from .fork import router as fork_router
from .analyze import router as analyze_router
from ..utils import MODEL_PATH, DATASET_LIST, TOKENIZER_LIST

api = APIRouter()
api.include_router(router)
api.include_router(analyze_router)
api.include_router(fork_router, prefix="/fork")

if not MODEL_PATH.exists():
    MODEL_PATH.mkdir(parents=True)


@api.get("/train/resources")
def get_datasets():
    return [DATASET_LIST, TOKENIZER_LIST]


@api.get("/models/options")
def get_model_options():
    return get_models_options("lm")


@api.get("/models")
def get_all_models(name: Optional[str] = None):
    """返回 MODEL_PATH 下的所有文件夹名，按创建时间倒序排序"""

    all_models = sorted(
        [model for model in MODEL_PATH.iterdir() if model.is_dir()],
        key=os.path.getctime,
        reverse=True,
    )
    # 如果有 name 参数，返回模糊查询的模型
    if name:
        all_models = [model for model in all_models if name in model.name]

    # 查找每个 model 的 checkpoint
    all_checkpoint = []
    for model_path in all_models:
        if (model_path / "config.json").exists():
            data = {"name": model_path.name, "config": {}, "checkpoints": []}

            # 读取每个 model_path 下的所有 external_config.json 文件
            external_config_path = model_path / "external_config.json"
            if external_config_path.exists():
                with open(external_config_path, "r") as f:
                    external_config = json.load(f)
                    data["config"] = external_config

            # 查找 model_path 下所有的 pth 后缀的文件
            for checkpoint in model_path.glob("*.pth"):
                data["checkpoints"].append(checkpoint.name)
            all_checkpoint.append(data)

    return all_checkpoint


@api.get("/models/{model_name}")
def get_model(model_name: str):
    model_path = MODEL_PATH / model_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

    model_config = model_path / "config.json"
    external_config = model_path / "external_config.json"
    exception_file = model_path / "exception.log"

    if model_config.exists() and external_config.exists():
        with open(model_config, "r") as f:
            model_config = json.load(f)
        with open(external_config, "r") as f:
            external_config = json.load(f)
        return {
            "model_name": model_name,
            "model_config": model_config,
            "external_config": external_config,
        }
    elif exception_file.exists():
        with open(exception_file, "r") as f:
            exception = f.read()
        raise HTTPException(status_code=400, detail=exception)


@router.post("/model/upload/weight")
async def upload_weight(name: str = Form(), file: UploadFile = File()):
    save_dir = MODEL_PATH / name
    if save_dir.exists():
        raise HTTPException(status_code=400, detail="Model name already exists")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not file.filename.endswith(".tar.gz"):
        raise HTTPException(status_code=400, detail="Only gzipped files are allowed")
    # ungizp upload folder gz to save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name

    with tarfile.open(tmp_file_path, mode="r:gz") as tar:
        top_folder = os.path.commonprefix(
            [
                member.name
                for member in tar.getmembers()
                if not member.name.startswith(".")
            ]
        )

        for member in tar.getmembers():
            member_path = os.path.relpath(member.name, top_folder)
            target_path = save_dir / member_path
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                with target_path.open("wb") as target:
                    extra_file = tar.extractfile(member)
                    if extra_file:
                        target.write(extra_file.read())
    return True


class DeleteBody(BaseModel):
    check_models: list[str]


@api.delete("/models/")
def delete_model(body: DeleteBody):
    check_models = body.check_models
    for model_name in check_models:
        model_path = MODEL_PATH / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        shutil.rmtree(model_path)

    return {"message": f"Model {model_name} deleted"}


async def read_csv(f, headers):
    rows = []
    async for line in f:
        if line and len(line) > 0:
            row = (
                zip(headers, line.strip().split(",")) if headers else []
            )  # 手动解析每一行
            row = {k: v for k, v in row}
            rows.append(row)
    return rows


async def read_train_csv(model_path: Path):
    pid_file_path = model_path / "pid"
    log_file_path = model_path / "train_log.csv"
    async with aiofiles.open(log_file_path, mode="r") as f:
        reader = csv.DictReader((await f.readline()).splitlines())  # 先解析 header 行
        headers = reader.fieldnames  # 获取列名
        rows = await read_csv(f, headers)
        if len(rows) > 0:
            yield json.dumps(rows)

        while pid_file_path.exists():  # 每次读取前检查 pid 文件是否存在
            await asyncio.sleep(1 / 30)
            rows = await read_csv(f, headers)
            if len(rows) > 0:
                yield json.dumps(rows)


@api.get("/models/{model_name}/log")
async def get_model_log(model_name: str):
    model_path = MODEL_PATH / model_name
    file_streaming = read_train_csv(model_path)

    return StreamingResponse(file_streaming, media_type="application/json")


@api.websocket("/models/{model_name}/logging")
async def get_model_logging(websocket: WebSocket):
    await websocket.accept()
    model_name: str = websocket.path_params["model_name"]
    if model_name is None:
        await websocket.send_text(f"EXCEPTION: Model name is required")
        await websocket.send_text("EOF")
        await websocket.close()
    else:
        model_path = MODEL_PATH / model_name
        exception_file_path = model_path / "exception.log"
        try:
            # 循环发送 read_train_csv 生成的数据
            train_csv = read_train_csv(model_path)
            async for log in train_csv:
                await websocket.send_text(log)
            if exception_file_path.exists():
                with open(exception_file_path, "r") as f:
                    exception = f.read()
                    await websocket.send_text(f"EXCEPTION: {exception}")
            else:
                await websocket.send_text("EOF")
            await websocket.close()
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except FileNotFoundError:
            print(f"Error: File not found")
            await websocket.close()


@api.post("/models/{model_name}/stop")
async def stop_train_model(model_name: str):
    model_path = MODEL_PATH / model_name
    pid_file_path = model_path / "pid"
    if not pid_file_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {model_name} not training")
    else:
        try:
            with open(pid_file_path, "r") as f:
                pid = int(f.read())
            os.kill(pid, signal.SIGTERM)
            os.remove(pid_file_path)
            return {"message": "Model training stopped"}
        except Exception as e:
            print(f"Error: {e}")
            os.remove(pid_file_path)
            return {"message": "Aleady stopped"}
