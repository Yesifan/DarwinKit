import os
import sys
import uvicorn
import multiprocessing
from pathlib import Path
from ..utils import DSPIKE_LLM_HOME

SERVER_LOG_PATH = Path(os.path.expanduser(DSPIKE_LLM_HOME)) / "server.log"


def start_uvicorn_server(port: int, daemon: bool):
    from ..server import app

    if daemon:
        sys.stdout = open(SERVER_LOG_PATH, "w")
        sys.stderr = sys.stdout
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print("uvicorn error:", e)


def start_uvicorn(port: int = 8000, daemon: bool = False) -> multiprocessing.Process:
    multiprocessing.set_start_method("spawn")
    server_thread = multiprocessing.Process(
        target=start_uvicorn_server, args=(port, daemon)
    )
    return server_thread
