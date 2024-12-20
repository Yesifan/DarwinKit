import click
from dataclasses import MISSING
from typing import Callable


def dataclass_options(options: dict, default=None) -> Callable:
    def decorator(func: Callable) -> Callable:
        # 自动生成 click 选项
        commands = []
        for name, field in options.items():
            option_default = default.__dict__[name] if default else field["default"]
            if option_default is MISSING:
                option_default = None

            if isinstance(field["type"], list):
                option_type = click.Choice(field["type"])
            else:
                option_type = eval(field["type"])

            # 添加 click 选项
            commands.append(
                click.option(
                    f"--{name}",
                    default=option_default,
                    type=option_type,
                    show_default=True,
                )
            )

        def wrapped_command(*args, **kwargs):
            nonlocal func
            # 获取 kwargs 中存在与 _fields 的参数
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in [name for name, _ in options.items()] and v is not None
            }

            # 把 config 添加到 args 的最后
            args = args + (relevant_kwargs,)
            return func(*args, **kwargs)

        # 保存原始函数的参数
        if hasattr(func, "__doc__"):
            wrapped_command.__doc__ = func.__doc__
        if hasattr(func, "__click_params__"):
            wrapped_command.__click_params__ = func.__click_params__

        for option in commands:
            wrapped_command = option(wrapped_command)

        return wrapped_command

    return decorator
