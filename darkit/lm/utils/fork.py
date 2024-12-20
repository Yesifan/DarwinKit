import torch
from pathlib import Path
from darkit.core.lib.fork_network import ForkNetowrk
from . import FORK_MODEL_PATH


class LMForkNetowrk(ForkNetowrk):
    def __init__(
        self,
        name: str,
        model: str = "",
        m_conf: dict = {},
        t_conf: dict = {},
        base_directory: Path = FORK_MODEL_PATH,
    ):
        super().__init__(name, model, m_conf, t_conf, base_directory)

    def create_network(self):
        from ..models import Metadata

        device = (
            self.t_conf.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )

        Model = Metadata[self.model]["cls"]
        mconf_cls = Metadata[self.model]["model"]
        tconf_cls = Metadata[self.model]["trainer"]

        mconf, tconf = mconf_cls(**self.m_conf), tconf_cls(**self.t_conf)

        model = Model(mconf).to(device)
        return model, mconf, tconf
