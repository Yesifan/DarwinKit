from pathlib import Path
from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles


app = FastAPI(docs_url="/api/docs", redoc_url=None)
api = APIRouter(prefix="/api")

try:
    from darkit.lm.server import router

    api.include_router(router, prefix="/lm")
    print("FastAPI Loaded LM server")
except ImportError as e:
    print("Error: lm.server not found", e)


SVELTE_DEV_SERVER = "http://localhost:5173"

app.include_router(api)

# 将 PWA 静态文件夹挂载到根路径
pwa_path = Path(__file__).parent / "web" / "build"
if pwa_path.exists():
    app.mount("/", StaticFiles(directory=pwa_path, html=True), name="pwa")
else:
    print(f"Warning: PWA build not found at {pwa_path}")
