import os
from pathlib import Path
from darkit.core.utils import DARWIN_KIT_HOME

PID_FILE = Path(os.path.expanduser(DARWIN_KIT_HOME)) / ".server.pid"


def save_pid(pid):
    with open(PID_FILE, "w") as f:
        f.write(str(pid))


def read_pid():
    try:
        with open(PID_FILE, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return None


def remove_pid_file():
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
