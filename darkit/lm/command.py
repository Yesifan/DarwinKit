import click
from pathlib import Path
from darkit.cli.src.decorator import dataclass_options
from darkit.core.utils import MODEL_PATH
from darkit.core.lib.options import get_models_options
from darkit.core.lib.analyze.nviz import load_model_from_addon
from .utils import MODEL_PATH, DATASET_LIST, TOKENIZER_LIST
from .utils.dataset import get_dataset

MODEL_FILE_DIR = Path(__file__).parent / "models"


@click.group(name="lm")
def command():
    """
    LM stands for Lanuage Model.
    """
    pass


@command.command()
def show():
    """
    Display the checkpoints of all models.
    """

    click.echo("TRAINED MODELS:")
    # 读取 MODEL_PATH 下的模型文件夹，输出模型名称
    for i, model in enumerate(MODEL_PATH.iterdir()):
        if model.is_dir():  # 排除__options__.json文件
            click.echo(f"  - {model.name}")
            # 模型文件夹下的每个 pth 文件都是一个版本的训练好的模型权重
            # 以 model:version 的形式输出每个版本
            for j, version in enumerate(model.iterdir()):
                if version.suffix == ".pth":
                    click.echo(f"    {j + 1}. {version.stem}")
    click.echo()


@command.command()
@click.argument(
    "MODEL_TYPE", type=click.Choice(["SpikeGPT", "RWKV_RNN", "SpikingLlama", "SpikeLM"])
)
@click.argument("MODEL_NAME", type=str)
@click.argument("PROMPT", type=str)
@click.option(
    "--device", type=click.Choice(["cpu", "cuda"]), default="cuda", show_default=True
)
@click.option(
    "--ctx_len",
    type=int,
    default=512,
    show_default=True,
    help="上下文长度。模型在处理输入序列时所能考虑的最大序列长度。",
)
def predict(model_type: str, model_name: str, prompt: str, device: str, ctx_len: int):
    """
    使用已经训练好的 SNN 大模型进行推理。
    可选的模型类型通过命令 darkit show model_types 查看。
    已经训练好的模型可通过命令 darkit show trained_models 查看, 使用时通过 MODEL_NAME:MODEL_VERSION 指定。

    Examples: darkit predict SpikeGPT SpikeGPT:complete "I am" --tokenizer gpt2 --ctx_len 512
    """
    import torch
    from darkit.core import Predicter

    # model_name = MODEL_NAME:MODEL_VERSION
    # 把 model_name 拆分为 model_name 和 version，如果没有 version 则默认为 'complete'
    checkpoint = None
    if ":" in model_name:
        model_name, checkpoint = model_name.split(":")

    with torch.no_grad():
        if model_type == "SpikeGPT" or model_type == "RWKV_RNN":
            from darkit.lm.models import SpikeGPT
        elif model_type == "SpikingLlama":
            from darkit.lm.models import SpikingLlama
        elif model_type == "SpikeLM":
            from darkit.lm.models import SpikeLM
        else:
            raise ValueError(f"Unknown model type {model_type}")

        predicter = Predicter.from_pretrained(
            name=model_name, checkpoint=checkpoint, device=device
        )
        predicter.predict(prompt, ctx_len)


@command.group()
@click.option(
    "--tokenizer",
    type=click.Choice(TOKENIZER_LIST),
    default=TOKENIZER_LIST[0] if TOKENIZER_LIST else "None",
)
@click.option(
    "--dataset",
    type=click.Choice(DATASET_LIST),
    default=DATASET_LIST[0] if DATASET_LIST else "None",
)
@click.option(
    "--resume",
    type=str,
    default=None,
)
@click.option(
    "--fork",
    type=str,
    default=None,
)
@click.pass_context
def train(ctx, tokenizer, dataset, resume, fork):
    """
    Train SNN Language Model.
    """
    from transformers import AutoTokenizer

    ctx.ensure_object(dict)
    tokenizer_name = tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataset = get_dataset(dataset)

    ctx.obj["dataset"] = train_dataset
    ctx.obj["tokenizer"] = tokenizer
    ctx.obj["resume"] = resume
    ctx.obj["fork"] = fork


def create_train_command(key: str, model_option: dict, trainer_option: dict):
    @click.command()
    @click.pass_context
    @dataclass_options(model_option)
    @dataclass_options(trainer_option)
    def train_model(ctx, m_conf: dict, t_conf: dict, **kwargs):
        """
        Train the specified model.
        """
        from .main import Trainer
        from .models import Metadata

        key_metadata = Metadata.get(key)

        if key_metadata is None:
            raise ValueError(f"Model {key} not found in options.json")
        print(f"Training model with type: {key}")

        Model = key_metadata["cls"]
        mconf_cls, tconf_cls = key_metadata["model"], key_metadata["trainer"]

        if mconf_cls is None and tconf_cls is None:
            raise ValueError(f"Model {key} not found in options.json")

        mconf, tconf = mconf_cls(**m_conf), tconf_cls(**t_conf)

        dataset, tokenizer = (ctx.obj["dataset"], ctx.obj["tokenizer"])

        mconf.vocab_size = tokenizer.vocab_size
        if key == "SpikeGPT":
            tconf.final_tokens = tconf.max_step

        model = Model(mconf).to(tconf.device)

        fork = ctx.obj["fork"]
        if fork is not None and fork != "":
            # load user modifications
            save_directory = MODEL_PATH / tconf.name
            addon_path = save_directory / "modification.json"
            if addon_path.exists():
                load_model_from_addon(model, addon_path)

        resume = ctx.obj["resume"]
        # with 语句确保了 Trainer 实例在进入和退出代码块时分别调用 __enter__ 和 __exit__ 方法
        # __enter__ 方法会调用 __init_pid___ ，在当前目录下生成一个 pid 文件，用于记录训练进程的 pid
        # __exit__ 方法会调用 __del_pid__，删除该文件
        with Trainer(
            model, config=tconf, tokenizer=tokenizer, resume=resume, fork=fork
        ) as trainer:
            try:
                trainer.train(dataset)
            except Exception as e:
                trainer.log_exception(e)

    return train_model


model_options = get_models_options("lm")
if model_options:
    for key, option in model_options.items():
        model_option = option["model"]
        trainer_option = option["trainer"]
        if model_option and trainer_option:
            train.add_command(
                create_train_command(key, model_option, trainer_option), key
            )
