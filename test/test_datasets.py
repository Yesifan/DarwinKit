import unittest
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from darkit.core.utils import dataset as dataset_utils
from darkit.core.utils.dataset.data_collator import MaskDataCollatorWithPadding


class TestTokenizedDataset(unittest.TestCase):
    def test_dataloader(self):
        # tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        wikitext = dataset_utils.create_tokenized_mask_dataset(
            wikitext,
            tokenizer=tokenizer,
        )
        wikitext_train = wikitext["train"]
        collect = MaskDataCollatorWithPadding(
            tokenizer=tokenizer, padding="max_length", max_length=100
        )
        dataloader = DataLoader(wikitext_train, batch_size=2, collate_fn=collect)  # type: ignore
        for data in dataloader:
            self.assertEqual(data["input_ids"].shape, data["labels"].shape)
            break

    def test_bert_dataset(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        wikitext = dataset_utils.create_tokenized_bert_dataset(
            wikitext,
            tokenizer=tokenizer,
        )
        wikitext_train = wikitext["validation"]
        collect = MaskDataCollatorWithPadding(
            tokenizer=tokenizer, padding="max_length", max_length=512
        )
        dataloader = DataLoader(wikitext_train, batch_size=24, collate_fn=collect)  # type: ignore
        for _ in tqdm(dataloader):
            pass

    def test_gpt_dataset(self):
        ctx_len = 1025
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        wikitext = wikitext["validation"]  # type: ignore
        datasetg = dataset_utils.create_tokenized_gpt_dataset(wikitext, tokenizer, ctx_len)  # type: ignore
        dataloader = DataLoader(datasetg, batch_size=2)
        for data in dataloader:
            self.assertEqual(data.shape[1], ctx_len)
            break

    def test_dataset(self):
        from datasets import load_dataset

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        ctx_len = 1024

        dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
        dataset = dataset["train"]  # type: ignore
        datasetg = dataset_utils.create_tokenized_gpt_dataset(dataset, tokenizer, ctx_len)  # type: ignore
        dataloader = DataLoader(datasetg, batch_size=2)

        for d in dataloader:
            print(d)
            break


if __name__ == "__main__":
    unittest.main()
