CLI_NAME = "darkit"


def dict_to_cmd_args(d: dict) -> str:
    return " ".join([f"--{k} {v}" for k, v in d.items() if v not in [None, ""]])


def gen_train_command(type: str, model: str, mconf: dict, tconf: dict):
    mconf_args = dict_to_cmd_args(mconf)
    tconf_args = dict_to_cmd_args(tconf)
    return f"{CLI_NAME} {type} train {model} {mconf_args} {tconf_args}"
