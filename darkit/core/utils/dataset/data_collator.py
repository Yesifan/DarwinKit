import torch
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from typing import Any, Dict, List, Union


class TokenizerDataCollator:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
    """

    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Any]):
        encode = self.tokenizer(
            features,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_ids = encode.input_ids

        context = input_ids[:, 1:-2]
        predict = input_ids[:, 2:-1]

        return [context, predict]


class LabelDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 加 1 是因为后面需要 slice 一下
        max_len = self.max_length + 1 if self.max_length else self.max_length
        encoded_inputs = {
            key: [example[key] for example in features] for key in features[0].keys()
        }
        for key, value in encoded_inputs.items():
            if isinstance(value[0], list):
                encoded_inputs[key] = [ids[:max_len] for ids in value]

        batch = self.tokenizer.pad(
            encoded_inputs,
            padding=self.padding,
            max_length=max_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        input_ids = batch.input_ids

        context = input_ids[:, 0:-1]
        predict = input_ids[:, 1:]

        return [context, predict]  # type: ignore


class MaskDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        encoded_inputs = {
            key: [example[key] for example in features]
            for key in features[0].keys()
            if key != "labels"
        }

        for key, value in encoded_inputs.items():
            if isinstance(value[0], list):
                encoded_inputs[key] = [ids[: self.max_length] for ids in value]

        batch = self.tokenizer.pad(
            encoded_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        mlm_labels = []
        max_length = self.max_length or 1000
        for example in features:
            labels = example["labels"]
            labels = labels[: self.max_length]
            exam_len = len(labels)
            mlm_labels.append(labels.copy() + [-100] * (max_length - exam_len))

        batch["labels"] = torch.tensor(mlm_labels)

        return batch  # type: ignore
