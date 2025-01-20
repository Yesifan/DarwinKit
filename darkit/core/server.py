import os
import json
import shutil
import tempfile
import traceback
import subprocess
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, APIRouter, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from .utils import BASE_MODEL_PATH
from .lib.logger import read_train_csv

from typing import Dict, Any, Optional


app = FastAPI(docs_url="/api/docs", redoc_url=None)
api = APIRouter(prefix="/api")

try:
    from darkit.lm.server import router

    api.include_router(router, prefix="/lm")
    print("FastAPI Loaded LM server")
except ImportError as e:
    print("Error: lm.server not found", e)


@api.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@api.get("/models")
def get_all_models(name: Optional[str] = None):
    """返回 MODEL_PATH 下的所有文件夹名，按创建时间倒序排序"""
    print("Getting all models", BASE_MODEL_PATH)
    if not BASE_MODEL_PATH.exists():
        return []
    all_models = sorted(
        [model for model in BASE_MODEL_PATH.iterdir() if model.is_dir()],
        key=os.path.getctime,
        reverse=True,
    )
    print("All models", all_models)
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
        else:
            print(f"Warning: Model {model_path} config.json not")

    return all_checkpoint


@api.get("/models/{model_name}")
def get_model(model_name: str):
    model_path = BASE_MODEL_PATH / model_name
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


class DeleteBody(BaseModel):
    check_models: list[str]


@api.delete("/models/")
def delete_model(body: DeleteBody):
    check_models = body.check_models
    for model_name in check_models:
        model_path = BASE_MODEL_PATH / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        shutil.rmtree(model_path)

    return {"message": f"Model {model_name} deleted"}


@api.get("/train/{model_name}/log")
async def get_model_log(model_name: str):
    model_path = BASE_MODEL_PATH / model_name
    file_streaming = read_train_csv(model_path)

    return StreamingResponse(file_streaming, media_type="application/json")


@api.websocket("/train/{model_name}/logging")
async def get_model_logging(websocket: WebSocket):
    await websocket.accept()
    model_name: str = websocket.path_params["model_name"]
    if model_name is None:
        await websocket.send_text(f"EXCEPTION: Model name is required")
        await websocket.send_text("EOF")
        await websocket.close()
    else:
        model_path = BASE_MODEL_PATH / model_name
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


########## Spiking ##########

model_file_cache = {}


@api.post("/spiking/upload")
async def upload_model(
    file: UploadFile = Form(...),
    cname: str = Form(...),
):
    from .lib.spiking import load_model_from_file
    from .lib.analyze import nviz

    if not file.filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="只支持上传 .py 文件")

    # 保存文件
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = Path(tmp_file.name)
        model_file_cache[cname] = tmp_file_path
        print(f"Saved uploaded file to {tmp_file_path}")
    # 动态加载模型
    try:
        model = load_model_from_file(cname, tmp_file_path)

        print("Loaded model", model)

        graph = nviz.get_model_graph(model)

        sub_graph = nviz.get_nx_sub_graph(graph)
        nx_nodes = list(sub_graph.nodes(data=True))
        nx_edges = list(sub_graph.edges())
        return [nx_nodes, nx_edges]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"加载模型失败: {e}")


current_task = None
current_task_name = None


class SpikingTrainModel(BaseModel):
    cname: str  # 代码中模型类的名称
    tconf: Dict[str, Any]
    sconf: Dict[str, Any]


@api.post("/spiking/train")
async def train_spiking_model(body: SpikingTrainModel):
    global current_task, current_task_name
    if current_task is not None:
        retcode = current_task.poll()
        if retcode is None:
            raise HTTPException(status_code=400, detail="Task is already running")
        else:
            print("Task finished with code", retcode)
            current_task = None
            current_task_name = None
    model_file = model_file_cache.get(body.cname)
    if model_file is None:
        raise HTTPException(status_code=400, detail="请先上传模型")
    tconf_str = json.dumps(body.tconf)
    sconf_str = json.dumps(body.sconf)
    command = f"darkit spiking train --file {model_file} --cname {body.cname} --tconf '{tconf_str}' --sconf '{sconf_str}'"
    print("Running command", command)
    current_task = subprocess.Popen(
        command,
        shell=True,
        text=True,
        bufsize=1,  # 使用行缓冲
        universal_newlines=True,  # 确保文本模式
    )
    current_task_name = body.tconf["name"]
    return {"status": "success", "message": "Task start running"}


app.include_router(api)

SVELTE_DEV_SERVER = "http://localhost:5173"

# 将 PWA 静态文件夹挂载到根路径
pwa_path = Path(__file__).parent / "web" / "build"
if pwa_path.exists():
    app.mount("/", StaticFiles(directory=pwa_path, html=True), name="pwa")
else:
    print(f"Warning: PWA build not found at {pwa_path}")
