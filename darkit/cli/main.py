import os
import signal
import click
from darkit import __version__
from .src.pid import read_pid, save_pid, remove_pid_file


@click.group()
@click.version_option(version=__version__, prog_name="DarwinKit")
def cli():
    pass


try:
    from darkit.lm.command import command as lm_command

    cli.add_command(lm_command)
except ImportError as e:
    print("lm command module not found", e)


@cli.command("start")
@click.option("--port", type=int, default=8000, help="Web 服务端口")
@click.option("--daemon", "-D", is_flag=True, help="是否以守护进程启动")
def start_server(port: int, daemon: bool):
    """
    开启 WEB 服务
    """
    from darkit.core.utils.server import start_uvicorn

    if read_pid():
        click.echo("服务已在运行。")
        return

    p = start_uvicorn(port, daemon)
    if daemon:
        p.start()
        save_pid(p.pid)
        print(f"服务已在后台以守护进程启动，端口: {port}")
        os._exit(0)
    else:
        print(f"服务已启动，端口: {port}")
        p.start()
        p.join()


@cli.command("stop")
def stop_server():
    """
    停止 WEB 服务
    """
    pid = read_pid()
    if pid is None:
        click.echo("服务未在运行或 PID 文件不存在。")
        return

    try:
        os.kill(pid, signal.SIGTERM)  # 发送终止信号
        remove_pid_file()  # 移除 PID 文件
        click.echo(f"服务已停止，PID: {pid}")
    except ProcessLookupError:
        click.echo(f"没有找到 PID 为 {pid} 的进程。")
        remove_pid_file()  # 移除 PID 文件以防错误
    except Exception as e:
        click.echo(f"停止服务时发生错误: {e}")


if __name__ == "__main__":
    cli()
