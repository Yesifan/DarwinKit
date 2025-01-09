# DarwinKit

## Introduction
This repository is a PyTorch-based natural language processing toolkit that provides some commonly used NLP models and tools.

## Getting Started For Users
You can read the following documents to start using this repository:
- [Tutorials](./docs/1.Introduction/1.About.md)
- [Installation](./docs/2.User-guide/1.Installation-guide.md)

## Getting Started For Developers

### Start the Project
- Start the FastAPI server
### Install Dependencies
We recommend using Anaconda as the Python environment manager. You can refer to the [installation guide](./docs/2.User-guide/1.Installation-guide.md#linux-with-anaconda) for steps to create a conda environment.
```bash
pip install -r requirements.txt
# requirements.txt does not include torch and torchvision
# Please install according to your environment, refer to the official documentation https://pytorch.org/get-started/locally/
# Example for CUDA 11.1:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Start the Project
- Start the fastapi server
  ```bash
  uvicorn core.server:app --reload --host 0.0.0.0
  ```
- Start the Svelte web
  ```bash
  cd core/web
  npm run dev
  ```
#### Notes
- When running `npm run dev`, pay attention to the `server.proxy` configuration in `vite.config.ts`, it should match the address of the `FastAPI` service you started.
- If you encounter errors like *Couldn't import the plugin "https://xxxxxxxxx.xxx/xxxxxxxxxxxx"*, it may be that your network cannot access `cdn.jsdelivr.net`. Please find a proxy or acceleration node and replace it in `project.inlang/settings.json`.

### Unit Testing
#### Writing Unit Tests
- Test naming conventions: Test files should start with test_, test classes should start with Test, and test methods should start with test_.
- Keep tests independent: Each test should run independently, ensuring that the execution of one test does not affect other tests.
- Cover various scenarios: Ensure tests cover normal cases, boundary cases, and exceptional cases.
- Use assertions: Use various assertion methods provided by unittest, such as assertEqual, assertTrue, assertFalse, etc.

#### Running Unit Tests
```bash
# All unit tests
python -m unittest discover
# Run unit tests for a specific file
python -m unittest test.model
```

### Packaging 
To create a distribution package (for uploading to PyPI or providing to users for installation), follow these steps:
1. Check if the version number in the `DarwinKit/__init__.py` file is correct.
2. Run `DarwinKit create-options` to update model information.
3. Check if the `frontend web static resources [DarwinKit/server/web/build]` folder exists. If not, build it according to the following steps:
    ```bash
    cd DarwinKit/server/web
    # Depends on node environment, if node is not installed, please install node first
    npm install # Install dependencies, no need to execute every time
    npm run build
    ```
4. Run the `python setup.py sdist bdist_wheel` command to build the distribution package.
5. After the build is complete, the distribution package is located in the `dist` folder. The file names of the distribution package are `DarwinKit-<version>.tar.gz` and `DarwinKit-<version>-py3-none-any.whl`.


## Third-Party Packages

This project includes third-party software. Below are the attributions and license details:

- [SpikeGPT](https://github.com/ridgerchu/SpikeGPT): Generative Pre-trained Language Model with Spiking Neural Networks
  - Copyright (c) 2023, Rui-Jie Zhu, Qihang Zhao, Guoqi Li, Jason K. Eshraghian
  - License: [BSD 2-Clause License](LICENSES/BSD-2-Clause-License.txt)
- [SpikeLM](https://github.com/Xingrun-Xing/SpikeLM): Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms
  - Copyright (c) 2024, Xingrun Xing, Boyan Gao, Zheng Zhang, David A. Clifton,
Shitao Xiao, Li Du, Guoqi Li, Jiajun Zhang
- SpikingLlama
  - Copyright (c), Zhiyuan Zhu, Qian Zheng

