import os
import torch
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from nnviz.entities import NNGraph

from darkit.core.utils import usetime
from darkit.core.lib.analyze import nviz, modanalyze
from ..utils import FORK_MODEL_PATH
from ..utils.fork import LMForkNetowrk

from typing import Optional, Literal


router = APIRouter()


class ModelTrainConfig(BaseModel):
    model: str
    m_conf: dict
    t_conf: dict


@router.post("/edit/init/{name}")
def visual_edit_init_model(name: str, config: ModelTrainConfig):
    """
    初始化模型配置并保存到指定路径。
    """
    try:
        fork = LMForkNetowrk(
            name=name,
            model=config.model,
            m_conf=config.m_conf,
            t_conf=config.t_conf,
        )
        fork.save()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class SharedData:
    def __init__(self):
        self.name: Optional[str] = None
        self.model: Optional[torch.nn.Module] = None
        self.graph: Optional[NNGraph] = None


shared_data = SharedData()


def get_shared_data():
    return shared_data


@router.get("/edit/load/{name}")
def visual_edit_load_model(name: str, shared: SharedData = Depends(get_shared_data)):
    """
    加载指定名称的模型及其配置，并生成模型的图结构。
    """

    if shared.graph is None:
        with usetime("load model"):
            fork_network = LMForkNetowrk.from_fork_name(name=name)
            model, _, _ = fork_network.create_network()
            shared.name = name
            shared.model = model
            shared.graph = nviz.get_model_graph(model)
    if shared.name != name:
        return JSONResponse(
            status_code=400,
            content={
                "code": 1,
                "name": shared.name,
                "detail": f"Aleady loaded model {shared.name}.",
            },
        )

    with usetime("get root sub graph"):
        sub_graph = nviz.get_nx_sub_graph(shared.graph)
    nx_nodes = list(sub_graph.nodes(data=True))
    nx_edges = list(sub_graph.edges())

    return {"edges": nx_edges, "nodes": nx_nodes}


class NetworkData(BaseModel):
    id: str
    nodes: list[str]


@router.post("/edit/subgraph")
async def get_sub_network(
    data: NetworkData, shared: SharedData = Depends(get_shared_data)
):
    """
    根据 id 获取指定节点的子图。
    """
    if shared.graph is None:
        raise HTTPException(status_code=400, detail="Model not loaded.")

    with usetime(f"get {data.id} sub graph"):
        sub_graph = nviz.get_nx_sub_graph(shared.graph, data.id, data.nodes)

    nx_nodes = list(sub_graph.nodes(data=True))
    nx_edges = list(sub_graph.edges())

    return {"edges": nx_edges, "nodes": nx_nodes}


@router.get("/edit/source")
async def get_source(
    id: Optional[str] = None, shared: SharedData = Depends(get_shared_data)
):
    """
    获取指定节点的源代码及其本地修改。

    Args:
        id (str): 节点 id, id使用点分隔路径。
    """
    if id is None:
        raise HTTPException(status_code=400, detail="module id not provided.")
    elif shared.model is None or shared.graph is None or shared.name is None:
        raise HTTPException(status_code=400, detail="Model not loaded.")

    save_directory = FORK_MODEL_PATH / shared.name
    module_path = save_directory / id.replace(".", "/")

    module = nviz.get_module_from_path(shared.model, id)
    if module is None:
        return []
    module_funcs = modanalyze.get_module_impl(module)

    for func in module_funcs:
        # 读取修改后保存到本地的代码
        func_file_path = module_path / (func["name"] + ".py")
        if func_file_path.exists():
            with open(func_file_path, "r") as file:
                func["body"] = file.read()

    return module_funcs


class EditData(BaseModel):
    module: str
    name: str
    code: str


@router.post("/edit/commit")
async def commit_changes(data: EditData, shared: SharedData = Depends(get_shared_data)):
    """
    提交对模型的修改。
    """
    if shared.name is None:
        raise HTTPException(status_code=400, detail="Model not loaded.")

    save_directory = FORK_MODEL_PATH / shared.name
    if not os.path.exists(save_directory):
        save_directory.mkdir(parents=True, exist_ok=True)

    module_name = data.module.replace(".", "/")
    module_path = save_directory / module_name

    if not module_path.exists():
        module_path.mkdir(parents=True, exist_ok=True)

    func_name = data.name + ".py"
    full_path = module_path / func_name

    with open(full_path, "w") as file:
        file.write(data.code)
    print(f"save {data.module}/{data.name} to {save_directory}")


@router.post("/edit/release")
async def release_model(data: dict, shared: SharedData = Depends(get_shared_data)):
    """
    释放实例化的模型。
    """
    if shared.name is None:
        raise HTTPException(status_code=400, detail="Model not loaded.")
    if data["name"] is not None and shared.name != data["name"]:
        raise HTTPException(status_code=400, detail="Model not match.")
    shared.name = None
    shared.model = None
    shared.graph = None
