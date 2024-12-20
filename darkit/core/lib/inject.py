import types
import torch
from pathlib import Path
from .analyze.nviz import get_module_from_path


def inject_script(model: torch.nn.Module, script_dir: Path):
    """
    遍历 script_dir 下的每一个文件夹，加载其中的代码并注入到模型中。
    然后如果 script_dir 还有文件夹则递归调用该方法。
    """

    def inject_code_from_path(model: torch.nn.Module, path: Path):
        # 查找路径下所有的 py 文件
        for file in path.glob("*.py"):
            # 获取父模块路径
            module_path = str(file.relative_to(script_dir).parent).replace("/", ".")
            module = get_module_from_path(model, module_path)
            func_name = file.stem
            with open(file, "r") as f:
                code = f.read()
                exec(code)
                new_func = locals().get(func_name)
                if new_func is not None:
                    setattr(module, func_name, types.MethodType(new_func, module))
        # 递归处理子文件夹
        for subdir in path.iterdir():
            if subdir.is_dir():
                inject_code_from_path(model, subdir)

    for subdir in script_dir.iterdir():
        if subdir.is_dir():
            inject_code_from_path(model, subdir)

    return model
