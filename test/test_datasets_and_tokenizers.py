import unittest
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from darkit.lm.utils import DATASET_LIST, TOKENIZER_LIST
from pathlib import Path
import shutil


class TestLoadDataset(unittest.TestCase):
    def setUp(self):
        self.test_dataset_list = DATASET_LIST
        self.temp_dataset_path = Path(__file__).parent / "test_datasets_temp"
        if not self.temp_dataset_path.exists():
            self.temp_dataset_path.mkdir()

        self.temp_dataset_path_str = str(self.temp_dataset_path)

    def tearDown(self):
        if self.temp_dataset_path.exists():
            shutil.rmtree(self.temp_dataset_path)

    def test_load_dataset_list(self):
        failed_dataset_num = 0
        failed_dataset_list: list[str] = []
        for dataset_name in self.test_dataset_list:
            path, name = None, None
            if ":" in dataset_name:
                path, name = dataset_name.split(":")
            else:
                path = dataset_name

            try:
                load_dataset(
                    path, name, split="train[:10]", cache_dir=self.temp_dataset_path_str
                )
            except Exception as e:
                failed_dataset_num += 1
                failed_dataset_list.append(dataset_name)
                print(f"Failed to load dataset {dataset_name}: {e}")

        self.assertEqual(
            failed_dataset_num,
            0,
            f"Failed to load {failed_dataset_num} dataset(s): {failed_dataset_list}",
        )


class TestAutoTokenizer(unittest.TestCase):
    def setUp(self):
        self.test_tokenizer_list = TOKENIZER_LIST

    def tearDown(self) -> None:
        return super().tearDown()

    def test_auto_tokenizer_list(self):
        failed_tokenizer_num = 0
        failed_tokenizer_list: list[str] = []
        for tokenizer_name in self.test_tokenizer_list:
            try:
                AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                failed_tokenizer_num += 1
                failed_tokenizer_list.append(tokenizer_name)
                print(f"Failed to load tokenizer {tokenizer_name}: {e}")

        self.assertEqual(
            failed_tokenizer_num,
            0,
            f"Failed to load {failed_tokenizer_num} tokenizer(s): {failed_tokenizer_list}",
        )


if __name__ == "__main__":
    unittest.main()
