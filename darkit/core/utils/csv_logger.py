import csv
from pathlib import Path
from typing import Any, Collection
from dataclasses import asdict


class CSVLogger:
    def __init__(self, filename: Path, fieldnames: Collection[Any]):
        self.filename = filename
        self.fieldnames = fieldnames
        self.file = open(filename, "a")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if self.file.tell() == 0:
            self.writer.writeheader()

    def log(self, data):
        if hasattr(data, "__dataclass_fields__"):
            data = asdict(data)
        self.writer.writerow(data)
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # Automatically close the file when the object is destroyed
    def __del__(self):
        self.close()
