import re
import torch
import inspect

banned_methods = [
    "register_buffer",
]


def get_module_functions(module: torch.nn.Module):
    """
    Get all the functions of a module.
    """
    src_list = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        print("func", name, func)
    return src_list


def get_module_impl(module):
    """
    获取 module 中所有的 self 函数
    从 __init__ 和 forward 开始递归获取模块中涉及的所有函数的信息。
    """
    src_list = []
    src_list += get_module_func_recursive(module, "__init__")
    src_list += get_module_func_recursive(module, "forward")
    return src_list


def get_module_func_recursive(module, func_name) -> list[dict[str, str]]:
    """
    获取 module 中指定函数的信息, 然后递归获取该函数中调用的所有 self 函数的信息。

    Args:
        module (object): 要分析的模块对象。
        func_name (str): 要获取的函数名称。

    Returns:
        list: 包含函数实现信息的字典列表。
    """
    assert hasattr(module, func_name), f"Module does not have function: {func_name}"
    src_list = []
    func_info = get_module_func(module, func_name)
    src_list.append(func_info)
    sub_method_list = extract_member_methods(func_info["body"])
    for sub_method in sub_method_list:
        if not hasattr(module, sub_method):
            continue
        func = getattr(module, sub_method)
        if not (callable(func) and inspect.ismethod(func)):
            continue
        if sub_method in banned_methods:
            continue
        src_list = src_list + get_module_func_recursive(module, sub_method)
    return src_list


def get_module_func(module, func_name: str):
    """
    获取给定模块中指定函数的函数名称、函数签名和函数主体。
    """
    func_body: str = inspect.getsource(getattr(module, func_name)).strip()

    return {"name": func_name, "body": func_body}


def extract_member_methods(func_body):
    """
    从给定的函数体中提取调用了的成员方法名称。

    此函数使用正则表达式查找所有出现的 “self.”，后跟提供的函数体字符串中的方法名称。它返回唯一方法名称的列表。

    Args:
        func_body (str): The body of the function as a string.

    Returns:
        list: A list of unique member method names found in the function body.
    """
    # Regular expression to match 'self.' followed by a method name
    pattern = r"self\.(\w+)\("
    # Find all occurrences of the pattern
    matches = re.findall(pattern, func_body)
    # Return unique method names
    return list(set(matches))
