from datasets import load_dataset, Dataset


def get_dataset(key, split="train") -> Dataset:
    path, name = None, None
    if ':' in key:
        path, name = key.split(':')
    else:
        path = key
    dataset = load_dataset(path, name, split=split, trust_remote_code=True)
    if dataset is None:
        raise ValueError(f"Unknown dataset: {key}")
    else:
        return dataset  # type: ignore