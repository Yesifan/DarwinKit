import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import Split, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from multiprocessing import Process

from . import packed_dataset


def create_gpt_dataset_process(
    dataset: Dataset,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    destination_path: Path,
    chunk_size: int,
    process_id: int = 0,
) -> None:
    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{process_id}",
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_token_id,  # type: ignore
        vocab_size=tokenizer.vocab_size,
        dtype="auto",
    )

    for text in tqdm(dataset):
        text = text["text"]  # type: ignore
        text_ids = tokenizer.encode(text)
        builder.add_array(np.array(text_ids, dtype=builder.dtype))


def create_gpt_dataset(
    dataset: Dataset,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    destination_path: Path,
    chunk_size: int,  # 该值需要与模型的 block_size 相关 (block_size+1) * (block_size/2)
    num_proc: int = 2,
) -> None:
    # 按照线程数分割数据集
    datasets = []
    for i in range(num_proc - 1):
        train_test = dataset.train_test_split(test_size=1 / (num_proc - i))
        datasets.append(train_test["test"])
        dataset = train_test["train"]
    datasets.append(dataset)  # 最后剩下的部分

    processes = []
    for i, subset in enumerate(datasets):
        p = Process(
            target=create_gpt_dataset_process,
            args=(
                subset,
                tokenizer,
                destination_path,
                chunk_size,
                i,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
