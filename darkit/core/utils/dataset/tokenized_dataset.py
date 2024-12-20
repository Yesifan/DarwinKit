import glob
from pathlib import Path
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_from_disk,
)
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .. import DATASET_PATH
from .packed import create_gpt_dataset, PackedDataset
from .pretreat import mask_single, create_next_sentence
from typing import Union, Optional, Union

PACKED_DATASET_PATH = DATASET_PATH / "tokenized_datasets"


def get_saved_path(
    dataset_name: str, tokenizer_name: str, external_path: Optional[str] = None
) -> Path:
    """
    Get the path to the packed datasets.
    """
    destination_path = PACKED_DATASET_PATH
    if external_path:
        destination_path = destination_path / external_path
    destination_path = destination_path / tokenizer_name / dataset_name  # type: ignore
    return destination_path


def create_tokenized_dataset(
    dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    num_proc: int = 2,
    **kwargs,
) -> Union[Dataset, DatasetDict]:
    """
    Tokenize the dataset and save it to disk.
    num_proc: 多进程处理数量
    """
    external_path = kwargs.get("external_path")
    tokenizer_name = tokenizer.name_or_path

    def encode_batch(batch):
        encode_fn = kwargs.get("encode_fn")
        if encode_fn:
            return encode_fn(batch)
        else:
            return tokenizer(batch["text"])

    if isinstance(dataset, Dataset):
        dataset_name = dataset.info.config_name or dataset.__class__.__name__
        destination_path = get_saved_path(dataset_name, tokenizer_name, external_path)
        destination_path = destination_path / str(dataset.split)
        if destination_path.exists():
            return load_from_disk(destination_path)

        tokenized_ds = dataset.map(
            encode_batch, batched=True, num_proc=num_proc, remove_columns="text"
        )
        tokenized_ds.save_to_disk(destination_path)
        return tokenized_ds

    elif isinstance(dataset, DatasetDict):
        dataset_dict = DatasetDict()
        ds = dataset["train"]
        dataset_name = ds.info.config_name or ds.__class__.__name__
        destination_path = get_saved_path(dataset_name, tokenizer_name, external_path)

        if destination_path.exists():
            return load_from_disk(destination_path)
        else:
            for split, ds in dataset.items():
                tokenized_ds = ds.map(
                    encode_batch, batched=True, num_proc=num_proc, remove_columns="text"
                )
                dataset_dict[split] = tokenized_ds

            dataset_dict.save_to_disk(destination_path)
            return dataset_dict
    else:
        raise TypeError(f"Dataset Type {dataset.__class__.__name__} is not support.")


def create_tokenized_mask_dataset(
    dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    num_proc: int = 2,
    **kwargs,
) -> Union[Dataset, DatasetDict]:
    def encode_fn(batch):
        batchEncoding = tokenizer(batch["text"])
        labels = []
        for ids in batchEncoding["input_ids"]:  # type: ignore
            _, label = mask_single(ids, tokenizer)
            labels.append(label)
        batchEncoding["labels"] = labels
        return batchEncoding

    return create_tokenized_dataset(
        dataset,
        tokenizer,
        num_proc,
        encode_fn=encode_fn,
        external_path="mask",
        **kwargs,
    )


def create_tokenized_bert_dataset(
    dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    num_proc: int = 2,
    **kwargs,
) -> Union[Dataset, DatasetDict]:
    def encode_fn(batch):
        sentences, next_sentence_labels = create_next_sentence(batch["text"])
        batchEncoding = tokenizer(sentences)

        labels = []
        for ids in batchEncoding["input_ids"]:  # type: ignore
            _, label = mask_single(ids, tokenizer)
            labels.append(label)
        batchEncoding["labels"] = labels
        batchEncoding["next_sentence_label"] = next_sentence_labels

        return batchEncoding
    return create_tokenized_dataset(
        dataset,
        tokenizer,
        num_proc,
        encode_fn=encode_fn,
        external_path="bert",
        **kwargs,
    )


def create_tokenized_gpt_dataset(
    dataset: Dataset,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ctx_len: int,
    num_proc: int = 2,
    **kwargs,
) -> PackedDataset:
    """
    Tokenize the dataset and save it to disk.
    num_proc: 多进程处理数量
    """
    external_path = "gpt"
    tokenizer_name = tokenizer.name_or_path

    dataset_name = dataset.info.config_name or dataset.__class__.__name__
    destination_path = get_saved_path(dataset_name, tokenizer_name, external_path)
    destination_path = destination_path / str(dataset.split)

    if destination_path.exists():
        print(f"Prepare Dataset destination path {destination_path} exists. Skip.")
    else:
        destination_path.mkdir(parents=True, exist_ok=True)
        default_chunk_size = 256 * 256 if len(dataset) > 256 * 256 else len(dataset)
        chunk_size = kwargs.get("chunk_size", default_chunk_size)
        create_gpt_dataset(dataset, tokenizer, destination_path, chunk_size, num_proc)

    filenames = sorted(glob.glob(str(destination_path / "*.bin")))
    n_chunks = 8 if len(filenames) > 8 else len(filenames)

    shuffle = kwargs.get("shuffle", False)
    process_rank = kwargs.get("process_rank", 0)
    num_processes = kwargs.get("num_processes", 1)
    return PackedDataset(
        filenames,
        n_chunks,
        block_size=ctx_len,
        shuffle=shuffle,
        process_rank=process_rank,
        num_processes=num_processes,
    )
