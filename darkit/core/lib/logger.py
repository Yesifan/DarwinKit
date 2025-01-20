import csv
import json
import asyncio
import aiofiles
from pathlib import Path


async def read_csv(f, headers):
    rows = []
    async for line in f:
        if line and len(line) > 0:
            row = (
                zip(headers, line.strip().split(",")) if headers else []
            )  # 手动解析每一行
            row = {k: v for k, v in row}
            rows.append(row)
    return rows


async def read_train_csv(model_path: Path):
    pid_file_path = model_path / "pid"
    log_file_path = model_path / "train_log.csv"
    async with aiofiles.open(log_file_path, mode="r") as f:
        reader = csv.DictReader((await f.readline()).splitlines())  # 先解析 header 行
        headers = reader.fieldnames  # 获取列名
        rows = await read_csv(f, headers)
        if len(rows) > 0:
            yield json.dumps(rows)

        while pid_file_path.exists():  # 每次读取前检查 pid 文件是否存在
            await asyncio.sleep(1 / 30)
            rows = await read_csv(f, headers)
            if len(rows) > 0:
                yield json.dumps(rows)
