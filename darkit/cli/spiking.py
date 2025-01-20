import json
import click
from pathlib import Path
from darkit.core.utils import BASE_DATASET_PATH
from darkit.core.lib.spiking import load_model_from_file, SpikingModule


class JsonType(click.ParamType):
    name = "json"

    def convert(self, value, param, ctx):
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            self.fail(f"{value} is not a valid JSON string", param, ctx)


@click.group(name="spiking")
def command():
    """
    脉冲化 ANN 模型
    """
    pass


@command.command()
@click.option("--file", type=str, default=8000, help="Web 服务端口")
@click.option("--cname", type=str, help="模型类型名称，指的是代码中模型类的名称")
@click.option("--tconf", type=JsonType(), help="训练器配置")
@click.option("--sconf", type=JsonType(), help="将模型脉冲化的配置")
def train(file, cname, tconf, sconf):
    """
    脉冲化 ANN 模型，并进行训练
    """
    import torchvision
    import torchvision.transforms as transforms
    from spikingjelly.activation_based import neuron, surrogate
    from darkit.core.trainer import Trainer, TrainerConfig

    model = load_model_from_file(cname, Path(file))
    for name, config in sconf.items():
        if not hasattr(model, name):
            raise click.ClickException(f"模型中不存在 {name} 层")
        if config["type"] == "LIF":
            module = neuron.IFNode(
                surrogate_function=surrogate.ATan(),
                step_mode="m",
                detach_reset=True,
                v_threshold=config["v_threshold"],
            )
        elif config["type"] == "IF":
            module = neuron.IFNode(
                surrogate_function=surrogate.ATan(),
                step_mode="m",
                detach_reset=True,
                v_threshold=config["v_threshold"],
            )
        else:
            raise click.ClickException(f"未知的神经元类型 {config['type']}")
        spiking_module = SpikingModule(tconf["T"], module)

        setattr(model, name, spiking_module)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        dataset = torchvision.datasets.FashionMNIST(
            root=BASE_DATASET_PATH, train=True, download=True, transform=transform
        )
        print("tconf", tconf)
        tconf = TrainerConfig(**tconf)
        with Trainer(model, config=tconf) as trainer:
            try:
                trainer.train(dataset)
            except Exception as e:
                trainer.log_exception(e)
