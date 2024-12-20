import os
from pathlib import Path
from darkit.core.utils import MODEL_PATH, DATASET_PATH

MODEL_PATH = Path(MODEL_PATH) / "lm"
FORK_MODEL_PATH = MODEL_PATH / "fork"
DATASET_PATH = Path(DATASET_PATH) / "lm"

DATASET_LIST = [
    "Salesforce/wikitext:wikitext-103-raw-v1",
    "Salesforce/wikitext:wikitext-103-v1",
    "Salesforce/wikitext:wikitext-2-raw-v1",
    "Salesforce/wikitext:wikitext-2-v1",
    "wikimedia/wikipedia:20231101.ab",
    "wikimedia/wikipedia:20231101.ace",
    "wikimedia/wikipedia:20231101.ady",
    "wikimedia/wikipedia:20231101.af",
    "HuggingFaceH4/ultrachat_200k",
    "HuggingFaceFW/fineweb:CC-MAIN-2013-20",
    "nthngdy/oscar-small:unshuffled_deduplicated_af",
]

TOKENIZER_LIST = [
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "google-bert/bert-base-uncased",
    "google-bert/bert-base-cased",
    "google-bert/bert-large-uncased",
    "google-bert/bert-large-cased",
    "google-bert/bert-base-multilingual-cased",
    "google-bert/bert-base-chinese",
]


def generate_model_options(model_name: str) -> Path:
    return MODEL_PATH / model_name
