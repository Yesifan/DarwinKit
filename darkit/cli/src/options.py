from dataclasses import is_dataclass, fields, MISSING
from enum import Enum
from typing import Type, Literal, Optional, Union, get_origin, get_args


def get_option_definer(conf_cls: Type, conf_comment: dict):
    """
    生成配置选项字典, 将 py 数据类型转换为 json 可序列化的数据类型。

    Args:
        conf_cls (Type): 需要是 dataclass 或者有 to_options 方法。to_options 方法返回一个字典，结构与 options 相同。
        conf_comment (dict): 配置字段的注释信息。

    Returns:
        dict: 配置选项字典。

    Raises:
        ValueError: 如果 conf_cls 不是 dataclass 且没有 to_options 方法。
    """
    has_method = hasattr(conf_cls, "to_options") and callable(
        getattr(conf_cls, "to_options")
    )
    if has_method:
        return conf_cls.to_options()
    elif is_dataclass(conf_cls):
        _fields = fields(conf_cls)

        options = dict()

        for field in _fields:
            option_default = field.default
            if option_default is MISSING:
                option_default = None

            option_type = field.type
            required = True
            if field.name == "device":
                option_type = Literal["cuda", "cpu"]
            if isinstance(option_type, type):
                if issubclass(option_type, Enum):  # type: ignore
                    all_values = [option.value for option in option_type]
                    option_type = all_values
                else:
                    option_type = option_type.__name__
            elif get_origin(option_type) is Union:
                option_type = "str"
            elif get_origin(option_type) is Literal:
                option_type = get_args(option_type)
            elif get_origin(option_type) is Optional:
                required = False
                option_type = get_args(option_type)[0]
            else:
                required = False
                option_type = "str"

            comment_dict = conf_comment.get(field.name, {})
            description = comment_dict.get("description")
            range = comment_dict.get("range")
            comment = f"{description} {range}" if description else None
            options[field.name] = {
                "default": option_default,
                "type": option_type,
                "required": required,
                "comment": comment,
            }

        return options
    else:
        raise ValueError(
            f"{conf_cls.__name__} should be a dataclass or have a to_options method."
        )
