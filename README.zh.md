# DarwinKit

## Introduction
本仓库是一个基于 PyTorch 的自然语言处理工具包，提供了一些常用的 NLP 模型和工具。

## Getting Started For Users
你可以阅读以下的文档，以开始使用本仓库：
- [Tutorials](./docs/1.Introduction/1.About.md)
- [Installation](./docs/2.User-guide/1.Installation-guide.md)

## Getting Started For Developers

### 安装依赖
我们建议使用 Anaconda 作为 Python 环境管理器。 创建 conda 环境的步骤可以参考[安装指引](./docs/zh/2.User-guide/1.Installation-guide.md#linux-with-anaconda)。
```bash
pip install -r requirements.txt
# requirements.txt 中未包含 torcu 和 torchvision
# 请根据自己的环境安装, 具体可查阅官方文档 https://pytorch.org/get-started/locally/
# Example for CUDA 11.1:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 启动项目
- Start the fastapi server
  ```bash
  uvicorn core.server:app --reload --host 0.0.0.0
  ```
- Start the svelte web
  ```bash
  cd core/web
  npm run dev
  ```
#### 注意事项
- 运行 `npm run dev` 时需要注意 `vite.config.ts` 中的 `server.proxy` 配置， 应该确保与你启动的 `FastAPI` 服务地址一致。
- 如果出现 *Couldn't import the plugin "https://cdn.jsdelivr.net/npm/@inlang/message-lint-rule-without-source@latest/dist/index.js"* 类似的错误，可能是你的网络无法访问 `cdn.jsdelivr.net`，请自行寻找代理或者加速节点并在 `project.inlang/settings.json` 中进行替换。

### 单元测试
#### 编写单元测试
- 测试命名规范：测试文件应以 test_ 开头，测试类应以 Test 开头，测试方法应以 test_ 开头。
- 保持测试独立：每个测试应独立运行，确保一个测试的执行不会影响其他测试。
- 覆盖多种情况：确保测试涵盖正常情况、边界情况和异常情况。
- 使用断言：使用 unittest 提供的各种断言方法，如 assertEqual、assertTrue、assertFalse 等。

#### 运行单元测试
```bash
# 所有单元测试
python -m unittest discover
# 运行指定文件的单元测试
python -m unittest test.model
```

### Packaging 
要创建分发包(用于上传到PyPI或提供给用户进行安装)，请按照以下步骤操作:
1. 检查 `DarwinKit/__init__.py` 文件中的版本号是否正确。
2. 运行 `DarwinKit create-options` 更新模型信息。
3. 检查 `前端网页静态资源[DarwinKit/server/web/build]` 文件夹是否存在，如果不存在则根据以下步骤进行构建:
    ```bash
    cd DarwinKit/server/web
    # 依赖 node 环境，如果没有安装 node，请先安装 node
    npm install # 安装依赖，不需要每次都执行
    npm run build
    ```
4. 运行 `python setup.py sdist bdist_wheel` 命令构建分发包。
5. 构建完成后，分发包的位置在 `dist` 文件夹下。分发包的文件名为 `DarwinKit-<version>.tar.gz` 和 `DarwinKit-<version>-py3-none-any.whl`。

## Third-Party

This project includes third-party software. Below are the attributions and license details:

- [SpikeGPT](https://github.com/ridgerchu/SpikeGPT/tree/master): Generative Pre-trained Language Model with Spiking Neural Networks
  - Copyright (c) 2023, Rui-Jie Zhu, Qihang Zhao, Guoqi Li, Jason K. Eshraghian
  - License: [BSD 2-Clause License](LICENSES/BSD-2-Clause-License.txt)