import os
import json
import copy
import glob
import types
import torch.nn as nn
import networkx as nx
from pathlib import Path
from nnviz.entities import NNGraph
from nnviz import inspection, drawing
from typing import List, Tuple, Optional


def get_model_graph(model: nn.Module, input=None) -> NNGraph:
    inspector = inspection.TorchFxInspector()
    graph = inspector.inspect(model, inputs=input)
    filter_tensor_constant_nodes(graph._graph)
    return graph


def filter_tensor_constant_nodes(graph: nx.MultiDiGraph):
    """
    从给定的有向多重图中移除所有节点名称以 `_tensor_constant` 开头的节点。
    """
    nodes_to_remove = [
        node for node in graph.nodes if node.startswith("_tensor_constant")
    ]
    for node in nodes_to_remove:
        graph.remove_node(node)


def nngraph_to_tree(nn_graph: nx.MultiDiGraph):
    """
    根据 nnviz 图中节点的 `path` 构建树，并以 `model.name` 作为树节点的名称。

    Args:
        nn_graph (nx.MultiDiGraph): nnviz 的 MultiDiGraph 图对象:
        nn_graph.nodes[0].data.model (nx.OpNodeModel): 节点的模型对象

    Returns:
        tree (dict): 树结构:
            {
                'input_ids': { 'name': 'input_ids', 'model': <OpNodeModel> },
                'bert': {
                    'children': {
                        'encoder': { 'name': 'encoder_1', 'model': <OpNodeModel> }
                    }
                }
            }
    """
    tree = {}

    for name, data in nn_graph.nodes(data=True):
        model = data.get("model")
        current_tree = tree
        segments: list = copy.deepcopy(model.path)
        segment = segments.pop(0)
        while segments:
            """
            逐层构建树结构，直到最后一个 segment 作为叶子节点保存 node 信息。
            """
            if segment not in current_tree:
                current_tree[segment] = {"children": {}}
            if "children" not in current_tree[segment]:
                current_tree[segment]["children"] = {}
            current_tree = current_tree[segment]["children"]
            segment = segments.pop(0)

        # 最后一个 segment 作为叶子节点保存 node 信息
        current_tree[segment] = {"name": name}

    return tree


def get_sub_tree(tree: dict, key: str):
    """
    获取指定键下的子树。

    Args:
        tree (dict): 树结构。
        key (str): 父节点键。E.g. 'bert.encoder'
    """
    subtree = tree
    segments = key.split(".")
    for segment in segments:
        if "children" in subtree:
            subtree = subtree["children"]
        subtree = subtree[segment]

    return subtree


def get_leaf_nodes(tree: dict, key: str):
    """
    获取指定键下的所有叶子节点 name 的 set。

    Args:
        tree (dict): 树结构。
        key (str): 父节点键。E.g. 'bert.encoder'

    Returns:
        set: 叶子节点的集合。
    """
    leaves = []
    subtree = get_sub_tree(tree, key)
    if "children" not in subtree:
        leaves.append(subtree["name"])
    else:
        for child_key in subtree["children"]:
            leaves.extend(get_leaf_nodes(subtree["children"], child_key))
    return set(leaves)


def rename_edges_to_parents(edges: List[Tuple[str, str]], nodes: List[str]):
    """
    将边列表中的节点名称重命名为其最靠近的祖先节点。 节点的名称格式为 `parent.child`, 代表节点的层级关系。
    Args:
        edges (List[Tuple[str, str]): 边列表
        nodes (List[str]): 现有的节点, 如果某条边的节点不在这个列表中, 则需要寻找其在这个列表中最靠近的祖宗节点
    """

    def get_parent(node: str):
        segments = node.split(".")
        while True:
            parent = ".".join(segments)
            if parent in nodes:
                return parent
            if not segments:
                return None
            segments.pop()

    renamed_edges = set()  # 使用 set 来去重
    for src, dst in edges:
        src = get_parent(src)
        dst = get_parent(dst)
        if src is None or dst is None:
            print("WARNING: edge not added", src, dst)
        else:
            renamed_edges.add((src, dst))  # 添加到 set 中
    return list(renamed_edges)  # 返回去重后的列表


def get_external_edge(graph: nx.MultiDiGraph, tree: dict, parent_key: str):
    """
    获取指定父节点的外部边。

    参数:
        G (nx.MultiDiGraph): 多重有向图。
        T (dict): 节点树，只有叶子节点是真实存在于 G 中的节点。
        parent_key (str): T 中的父节点键。

    返回:
        List[Tuple[str, str]]: 外部边的列表。
    """

    leaf_nodes = get_leaf_nodes(tree, parent_key)

    # 获取外部边
    external_in_edges = []
    external_out_edges = []
    for node_key in leaf_nodes:
        node_name = graph.nodes[node_key]["model"].name
        for _, neighbor in graph.out_edges(node_key):
            if neighbor not in leaf_nodes:
                neighbor_name = graph.nodes[neighbor]["model"].name
                external_out_edges.append((node_name, neighbor_name))
        for neighbor, _ in graph.in_edges(node_key):
            if neighbor not in leaf_nodes:
                neighbor_name = graph.nodes[neighbor]["model"].name
                external_in_edges.append((neighbor_name, node_name))

    return external_in_edges, external_out_edges


def get_nx_sub_graph(
    nx_graph: NNGraph, key: Optional[str] = None, nodes: List[str] = []
):
    """
    获取给定节点中指定 key 下的第一层节点, 如果节点有子节点则将其所有节点视为一个节点。

    Args:
        nx_graph (NNGraph): nnviz 图对象
        key (str): 父节点键
        nodes (List[str]): ���有的节点, 如果某条��的节点不在这个列表中, 则需要寻找其在这个列表中最靠近的祖宗节点
    """
    graph = nx_graph._graph
    graph_tree = nngraph_to_tree(graph)

    sub_graph = nx.MultiDiGraph()
    if key is None:
        top_modules_key = list(graph_tree.keys())
    else:
        # 将需要扩展的节点从 nodes 中去除
        nodes = [node for node in nodes if node != key]
        sub_graph_tree = get_sub_tree(graph_tree, key).get("children", {})
        top_modules_key = [f"{key}.{k}" if key else k for k in sub_graph_tree.keys()]

    nodes = top_modules_key + nodes

    for sub_module_key in top_modules_key:
        sub_module_size = len(
            get_sub_tree(graph_tree, sub_module_key).get("children", {})
        )
        sub_graph.add_node(
            sub_module_key,
            sub_module_size=sub_module_size,
        )

        in_edges, out_edges = get_external_edge(graph, graph_tree, sub_module_key)

        renamed_edges = rename_edges_to_parents(in_edges + out_edges, nodes)  # type: ignore
        # 过滤掉错误的边
        renamed_edges = [
            edge for edge in renamed_edges if not sub_graph.has_edge(*edge)
        ]
        sub_graph.add_edges_from(renamed_edges)

    return sub_graph


def get_module_from_path(module: nn.Module, path: str) -> Optional[nn.Module]:
    """
    使用点分隔路径从给定模块中检索子模块。
    """
    module_keys = path.split(".")
    for key in module_keys:
        if hasattr(module, key):
            module = getattr(module, key)
        else:
            return None
    return module


def load_model_from_addon(model: nn.Module, addon_path: str):
    with open(addon_path, "r") as f:
        addon_dict = json.load(f)
        for module_path, attr_list in addon_dict.items():
            module = get_module_from_path(model, module_path)
            for attr, code in attr_list.items():
                exec(code)
                if attr != "init":
                    new_func = locals().get(attr)
                    if new_func is not None:
                        setattr(module, attr, types.MethodType(new_func, module))


def dynamic_code_injection(model: nn.Module, path: Path):
    """
    从给定路径加载模块的动态代码，并将其注入到模型中。
    """
    py_files = glob.glob(os.path.join(path, "**", "*.py"), recursive=True)
    for file in py_files:
        file = Path(file)
        module_path = file.parent.name.replace("_", ".")
        module = get_module_from_path(model, module_path)
        func_name = file.name.replace(".py", "")
        with open(file, "r") as f:
            code = f.read()
            exec(code)
            new_func = locals().get(func_name)
            if new_func is not None:
                setattr(module, func_name, types.MethodType(new_func, module))
