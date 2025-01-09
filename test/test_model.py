import random
import shutil
import unittest
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Tokenizer
from darkit.lm.main import Trainer, Predicter


# python -m unittest test.test_model.TestSpikeGPT
class TestSpikeGPT(unittest.TestCase):
    model_name = f"GPT-Test-Train-{random.randint(1000, 9999)}"
    # model_name = "GPT-Test-Train-8227"

    def test_train_predict(self):
        from darkit.lm.models.SpikeGPT import SpikeGPT, SpikeGPTConfig, TrainerConfig

        device = "cuda"
        ctx_len = 64

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        wikitext_train = wikitext["train"]  # type: ignore

        config = SpikeGPTConfig(
            tokenizer.vocab_size,
            ctx_len=ctx_len,
            model_type="RWKV",
            n_layer=12,
            n_embd=768,
        )
        model = SpikeGPT(config)
        tconf = TrainerConfig(
            name=self.model_name,
            device=device,
            max_epochs=1,
            epoch_length_fixed=100,
            batch_size=2,
            save_step_interval=1,
        )
        with Trainer(model, tokenizer=tokenizer, config=tconf) as trainer:
            trainer.train(train_dataset=wikitext_train)

        self.assertTrue(
            trainer.is_name_exist,
            f"Model {self.model_name} not saved.",
        )

        # 测试模型
        predicter = Predicter.from_pretrained(self.model_name)
        prompt = "hello world"
        print(prompt, end="")
        for char in predicter.predict(prompt, ctx_len=ctx_len):
            print(char, end="", flush=True)
        print()

        # 测试 RWKV_RNN
        from darkit.lm.models.SpikeGPT.predicter import RWKVRNNPredicter

        rwkv_predicter = RWKVRNNPredicter.from_pretrained(self.model_name)
        print(prompt, end="")
        for char in rwkv_predicter.predict(prompt, ctx_len=ctx_len):
            print(char, end="", flush=True)
        print()

        # 删除模型
        if trainer.save_directory:
            shutil.rmtree(trainer.save_directory)
            print(f"Model {self.model_name} deleted.")


# # python -m unittest test.test_model.TestSpikingLlama
# class TestSpikingLlama(unittest.TestCase):
#     model_name = f"SpikingLlama-Test-{random.randint(1000, 9999)}"
#     # model_name = "SpikingLlama-Test-Train-7249"

#     def test_train_predict(self):
#         from darkit.lm.models.SpikingLlama import (
#             SpikingLlama,
#             SpikingLlamaConfig,
#             TrainerConfig,
#         )

#         block_size = 128
#         tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#         config = SpikingLlamaConfig(
#             org="StatNLP-research",
#             name=self.model_name,
#             block_size=block_size,
#             vocab_size=tokenizer.vocab_size,
#             padding_multiple=64,
#             n_layer=12,
#             n_head=12,
#             n_embd=768,
#             rotary_percentage=1.0,
#             parallel_residual=False,
#             bias=False,
#             _norm_class="FusedRMSNorm",  # type: ignore
#             norm_eps=1e-5,
#             _mlp_class="LLaMAMLP",
#             intermediate_size=2048,
#             n_query_groups=1,
#         )

#         wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
#         wikitext_train = wikitext["train"]  # type: ignore
#         wikitext_validation = wikitext["validation"]  # type: ignore
#         model = SpikingLlama(config)
#         tconfig = TrainerConfig(
#             name=self.model_name,
#             max_step=100,
#             save_step_interval=100,
#             eval_step_interval=100,
#         )

#         trainer = Trainer(model, tokenizer=tokenizer, config=tconfig)
#         trainer.train(wikitext_train, wikitext_validation)

#         self.assertTrue(
#             trainer.is_name_exist,
#             f"Model {self.model_name} not saved.",
#         )

#         # 测试模型
#         ctx_len = 64
#         predicter = Predicter.from_pretrained(self.model_name)
#         prompt = "hello world"
#         print(prompt, end="")
#         for char in predicter.predict(prompt, ctx_len=ctx_len):
#             print(char, end="", flush=True)
#         print()

#         # 删除模型
#         if trainer.save_directory:
#             shutil.rmtree(trainer.save_directory)
#             print(f"Model {self.model_name} deleted.")


class TestSpikeLM(unittest.TestCase):
    model_name = f"SpikeLM-Test-{random.randint(1000, 9999)}"
    # model_name = "SpikeLM-Test-1417"

    def test_train_predict(self):
        from darkit.lm.models.SpikeLM import TrainerConfig, SpikeLMConfig, SpikeLM

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        m_conf = SpikeLMConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=72,
            num_hidden_layers=6,
            num_attention_heads=6,
        )
        model = SpikeLM(m_conf)

        t_conf = TrainerConfig(
            name=self.model_name,
            batch_size=2,
            max_train_steps=10,
        )

        # 训练模型
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        wikitext_train = wikitext["train"]  # type: ignore

        with Trainer(model, tokenizer=tokenizer, config=t_conf) as trainer:
            trainer.train(wikitext_train)

        self.assertTrue(
            trainer.is_name_exist,
            f"Model {self.model_name} not saved.",
        )

        # 测试模型
        ctx_len = 64
        predicter = Predicter.from_pretrained(self.model_name)
        prompt = "hello world"
        print(prompt, end="")
        for char in predicter.predict(prompt, ctx_len=ctx_len):
            print(char, end="", flush=True)
        print()

        # 删除模型
        if trainer.save_directory:
            shutil.rmtree(trainer.save_directory)
            print(f"Model {self.model_name} deleted.")


# python -m unittest test.test_model.TestTrainerException
class TestTrainerException(unittest.TestCase):
    model_name = f"GPT-Test-Train-{random.randint(1000, 9999)}"

    def test_train_predict(self):
        from darkit.lm.models.SpikeGPT import SpikeGPT, SpikeGPTConfig, TrainerConfig

        device = "cuda"

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        wikitext_train = wikitext["train"]  # type: ignore

        config = SpikeGPTConfig(tokenizer.vocab_size)
        model = SpikeGPT(config)
        tconf = TrainerConfig(name=self.model_name, device=device)
        try:
            with Trainer(model, tokenizer=tokenizer, config=tconf) as trainer:
                trainer.train(train_dataset=wikitext_train)
        except Exception as e:
            trainer.log_exception(e)

        self.assertTrue((trainer.save_directory / "exception.log").exists())  # type: ignore
        print(f"Exception log saved to {trainer.save_directory / 'exception.log'}")  # type: ignore

        # 删除模型
        if trainer.save_directory:
            shutil.rmtree(trainer.save_directory)
            print(f"Model {self.model_name} deleted.")


# python -m unittest test.test_model.TestTrainerResume
class TestTrainerResume(unittest.TestCase):
    model_name = f"GPT-Test-Train-{random.randint(1000, 9999)}"

    def test_train_predict(self):
        from darkit.lm.models.SpikeGPT import SpikeGPT, SpikeGPTConfig, TrainerConfig

        device = "cuda"
        ctx_len = 64

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        wikitext_train = wikitext["train"]  # type: ignore

        config = SpikeGPTConfig(
            tokenizer.vocab_size,
            ctx_len=ctx_len,
            model_type="RWKV",
            n_layer=12,
            n_embd=768,
        )
        model = SpikeGPT(config)
        tconf = TrainerConfig(
            name=self.model_name,
            device=device,
            batch_size=2,
            max_step=100,
            save_step_interval=100,
        )
        with Trainer(model, tokenizer=tokenizer, config=tconf) as trainer:
            trainer.train(train_dataset=wikitext_train)

        self.assertTrue(
            trainer.is_name_exist,
            f"Model {self.model_name} not saved.",
        )

        # 测试模型
        predicter = Predicter.from_pretrained(self.model_name)
        prompt = "hello world"
        print(prompt, end="")
        for char in predicter.predict(prompt, ctx_len=ctx_len):
            print(char, end="", flush=True)
        print()

        # 恢复训练
        print("Resuming training...")
        model = SpikeGPT(config)
        tconf2 = TrainerConfig(
            name=self.model_name,
            device=device,
            batch_size=2,
            max_step=200,
            save_step_interval=100,
        )
        with Trainer(
            model, tokenizer=tokenizer, config=tconf2, resume=self.model_name
        ) as trainer:
            trainer.train(train_dataset=wikitext_train)

        # 测试模型
        predicter = Predicter.from_pretrained(self.model_name)
        prompt = "hello world"
        print(prompt, end="")
        for char in predicter.predict(prompt, ctx_len=ctx_len):
            print(char, end="", flush=True)
        print()

        # 删除模型
        if trainer.save_directory:
            shutil.rmtree(trainer.save_directory)
            print(f"Model {self.model_name} deleted.")


if __name__ == "__main__":
    unittest.main()
