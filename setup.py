"""
python setup.py sdist bdist_wheel
python -m twine upload dist/*
"""

import os
import subprocess
from setuptools import setup, find_packages
from darkit.core.utils import PWA_PATH

requirements = ["torch"]


def read_version():
    with open(os.path.join("darkit", "__init__.py")) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"')
    return "0.0.1"


if not PWA_PATH.exists():
    print(f"Warning: PWA build not found at {PWA_PATH}")
    exit(1)

# 执行 init.py 文件
subprocess.run(["python", "init.py"], check=True)

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()

with open("./README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="DarwinKit",
    version=read_version(),
    author="ZJU, and other contributors",
    author_email="yesifan66@zju.edu.cn",
    description="A deep learning framework for SNNs built on PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zju-bmi-lab/DarwinKit",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.9",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Spiking Neural Network",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "darkit=darkit.cli.main:cli",
        ],
    },
)
