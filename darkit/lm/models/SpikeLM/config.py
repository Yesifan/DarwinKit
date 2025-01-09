from dataclasses import dataclass, field
from transformers import SchedulerType
from transformers.models.bert.configuration_bert import BertConfig
from darkit.core import TrainerConfig as BaseTrainerConfig
from typing import Optional


@dataclass
class SpikeLMConfig(BertConfig):
    # SpikeLMConfig
    weight_bits: int = 32  # 权重量化位数
    input_bits: int = 2  # for bidirectional spike with level {-1,0,1}
    clip_init_val: float = 2.5
    weight_layerwise: bool = True
    input_layerwise: bool = True
    hidden_act: str = "relu"  # 输出层激活函数
    num_hidden_layers: int = 12  # Transformer Block的层数
    quantize_act: bool = True
    clip_val: float = 1.0  # 神经元阈值初始值
    T: int = 4  # SNN时间步长

    # BertConfig
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    # pad_token_id: int = 0
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    classifier_dropout: Optional[float] = None

    # PretrainedConfig
    return_dict: bool = True
    output_hidden_states: bool = False
    output_attentions: bool = False
    torchscript: bool = False
    torch_dtype: Optional[str] = None
    use_bfloat16: bool = False
    tf_legacy_loss: bool = False
    pruned_heads: dict = field(default_factory=dict)
    tie_word_embeddings: bool = True
    chunk_size_feed_forward: int = 0
    is_encoder_decoder: bool = False
    is_decoder: bool = False
    cross_attention_hidden_size: Optional[int] = None
    add_cross_attention: bool = False
    tie_encoder_decoder: bool = False

    architectures: Optional[str] = None
    finetuning_task: Optional[str] = None
    id2label: Optional[dict] = None
    label2id: Optional[dict] = None

    tokenizer_class: Optional[str] = None
    prefix: Optional[str] = None
    bos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    sep_token_id: Optional[int] = None
    decoder_start_token_id: Optional[int] = None
    task_specific_params: Optional[dict] = None

    _name_or_path: str = ""
    _commit_hash: Optional[str] = None
    _attn_implementation_internal: Optional[str] = None
    transformers_version: Optional[str] = None

    max_position_embeddings: int = 512

    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: Whether or not return [`~utils.ModelOutput`] instead of tuples.
        """
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict and not self.torchscript

    @property
    def num_labels(self) -> int:
        """
        `int`: The number of labels for classification models.
        """
        return len(self.id2label) if self.id2label else 0

    @num_labels.setter
    def num_labels(self, num_labels: int):
        if (
            not hasattr(self, "id2label")
            or self.id2label is None
            or len(self.id2label) != num_labels
        ):
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    @property
    def _attn_implementation(self):
        # This property is made private for now (as it cannot be changed and a PreTrainedModel.use_attn_implementation method needs to be implemented.)
        if hasattr(self, "_attn_implementation_internal"):
            if self._attn_implementation_internal is None:
                # `config.attn_implementation` should never be None, for backward compatibility.
                return "eager"
            else:
                return self._attn_implementation_internal
        else:
            return "eager"

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        self._attn_implementation_internal = value


@dataclass
class TrainerConfig(BaseTrainerConfig):
    """
    max_seq_length: The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.
    """

    name: str = "SpikeLM"
    device: str = "cuda"
    device_num: int = 1
    learning_rate: float = 5e-5
    max_train_steps: int = 100000
    # 标记化后的最大总输入序列长度。长于此的序列将被截断。
    max_seq_length: int = 512
    batch_size: int = 1
    val_batch_size: int = 1
    lr_scheduler_type: SchedulerType = SchedulerType.LINEAR
    weight_decay: float = 1e-2
    num_warmup_steps: int = 0
    gradient_accumulation_steps: int = 1

    save_step_interval: int = 1000


SPIKELM_CONFIG_COMMENT = {
    # SpikeLMConfig related configurations
    "weight_bits": {
        "description": "Weight quantization bit width. This determines the number of bits used to represent weights in the model.",
        "range": "Typically between 1 and 32, with common values being 8, 16, or 32",
    },
    "input_bits": {
        "description": "Input quantization bit width. This determines the number of bits used to represent input data, typically for bidirectional spike with levels {-1, 0, 1}.",
        "range": "Typically 1 or 2, depending on the input representation",
    },
    "clip_init_val": {
        "description": "Initial value for the clipping threshold. This is used to clip the activations during training.",
        "range": "Positive float, typically between 1.0 and 5.0",
    },
    "weight_layerwise": {
        "description": "Whether to apply layer-wise weight quantization. If True, each layer will have its own quantization parameters.",
        "range": "Boolean, True or False",
    },
    "input_layerwise": {
        "description": "Whether to apply layer-wise input quantization. If True, each layer will have its own quantization parameters.",
        "range": "Boolean, True or False",
    },
    "hidden_act": {
        "description": "Activation function used in the hidden layers. Common choices include 'relu', 'gelu', etc.",
        "range": "'relu', 'gelu', 'tanh', 'sigmoid', etc.",
    },
    "num_hidden_layers": {
        "description": "Number of hidden layers (Transformer Blocks) in the model.",
        "range": "Positive integer, typically between 6 and 24",
    },
    "quantize_act": {
        "description": "Whether to quantize the activations. If True, activations will be quantized during training and inference.",
        "range": "Boolean, True or False",
    },
    "clip_val": {
        "description": "Threshold value for neuron activation. This is used to determine when a neuron fires.",
        "range": "Positive float, typically between 0.5 and 2.0",
    },
    "T": {
        "description": "Number of time steps for the Spiking Neural Network (SNN). This determines the temporal resolution of the SNN.",
        "range": "Positive integer, typically between 1 and 10",
    },
    # BertConfig related configurations
    "vocab_size": {
        "description": "Vocabulary size. This represents the number of different tokens the model can handle.",
        "range": "Typically between 30,000 and 50,000, e.g., 30,522 for BERT",
    },
    "hidden_size": {
        "description": "Dimensionality of the encoder layers and the pooler layer. This is also the embedding dimension.",
        "range": "Common values are 768, 1024, 1280, etc., depending on the model scale",
    },
    "num_attention_heads": {
        "description": "Number of attention heads for each attention layer in the Transformer blocks.",
        "range": "Positive integer, typically 12, 16, 20, etc., depending on the model scale",
    },
    "intermediate_size": {
        "description": "Dimensionality of the feed-forward layers in the Transformer blocks.",
        "range": "Common values are 3072, 4096, etc., depending on the model scale",
    },
    "hidden_dropout_prob": {
        "description": "Dropout probability for the hidden states in the Transformer blocks.",
        "range": "Float between 0.0 and 0.5, typically 0.1",
    },
    "attention_probs_dropout_prob": {
        "description": "Dropout probability for the attention probabilities in the Transformer blocks.",
        "range": "Float between 0.0 and 0.5, typically 0.1",
    },
    "max_position_embeddings": {
        "description": "Maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).",
        "range": "Positive integer, typically 512, 1024, 2048, etc.",
    },
    "type_vocab_size": {
        "description": "The vocabulary size of the token_type_ids passed into BertModel. For BERT, this is 2 (sentence A and sentence B).",
        "range": "Positive integer, typically 2",
    },
    "initializer_range": {
        "description": "Standard deviation of the truncated_normal_initializer for initializing all weight matrices.",
        "range": "Small positive float, typically 0.02",
    },
    "layer_norm_eps": {
        "description": "Epsilon value for the Layer Normalization layers.",
        "range": "Very small positive float, typically 1e-12",
    },
    "position_embedding_type": {
        "description": "Type of position embeddings to use. Options include 'absolute' and 'relative'.",
        "range": "'absolute', 'relative'",
    },
    "use_cache": {
        "description": "Whether or not the model should return the last key/values attentions (not used by all models).",
        "range": "Boolean, True or False",
    },
    "classifier_dropout": {
        "description": "The dropout ratio for the classification head.",
        "range": "Float between 0.0 and 0.5, or None if no dropout is applied",
    },
    # PretrainedConfig related configurations
    "return_dict": {
        "description": "Whether to return a `~utils.ModelOutput` instead of a plain tuple.",
        "range": "Boolean, True or False",
    },
    "output_hidden_states": {
        "description": "Whether to return the hidden states of all layers. Can be used to initialize the corresponding layers in a downstream task.",
        "range": "Boolean, True or False",
    },
    "output_attentions": {
        "description": "Whether to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.",
        "range": "Boolean, True or False",
    },
    "torchscript": {
        "description": "Whether to enable TorchScript support. Setting this to True will force `return_dict=False` to avoid JIT errors.",
        "range": "Boolean, True or False",
    },
    "torch_dtype": {
        "description": "The desired data type of the parameters and buffers in this module, e.g., 'float32' or 'float16'.",
        "range": "String representing a data type, e.g., 'float32', 'float16', or None",
    },
    "use_bfloat16": {
        "description": "Whether to use bfloat16 precision for the model.",
        "range": "Boolean, True or False",
    },
    "tf_legacy_loss": {
        "description": "Whether to use the legacy TensorFlow loss calculation. This is only relevant for compatibility with older TensorFlow models.",
        "range": "Boolean, True or False",
    },
    "pruned_heads": {
        "description": "Dictionary of pruned heads. Keys are layer indices and values are lists of heads to prune in that layer.",
        "range": "Dictionary, e.g., {1: [0, 1], 2: [2, 3]}",
    },
    "tie_word_embeddings": {
        "description": "Whether to tie the input and output word embeddings. This is useful for language modeling tasks.",
        "range": "Boolean, True or False",
    },
    "chunk_size_feed_forward": {
        "description": "Chunk size for the feed-forward layers. This can be used to reduce memory usage during training.",
        "range": "Non-negative integer, typically 0 (no chunking) or a positive integer",
    },
    "is_encoder_decoder": {
        "description": "Whether the model is an encoder-decoder model. If True, the model will have both an encoder and a decoder.",
        "range": "Boolean, True or False",
    },
    "is_decoder": {
        "description": "Whether the model is a decoder-only model. If True, the model will only have a decoder.",
        "range": "Boolean, True or False",
    },
    "cross_attention_hidden_size": {
        "description": "The hidden size of the cross-attention layers. This is used in encoder-decoder models.",
        "range": "Positive integer or None (if not applicable)",
    },
    "add_cross_attention": {
        "description": "Whether to add cross-attention layers to the model. This is used in encoder-decoder models.",
        "range": "Boolean, True or False",
    },
    "tie_encoder_decoder": {
        "description": "Whether to tie the weights of the encoder and decoder. This is useful for seq2seq tasks.",
        "range": "Boolean, True or False",
    },
    # TrainerConfig related configurations
    "name": {
        "description": "Name of the trainer, used to identify different training instances.",
        "range": "Custom string",
    },
    "device": {
        "description": "Device used for training, e.g., 'cuda' for GPU or 'cpu' for CPU.",
        "range": "'cuda', 'cpu'",
    },
    "device_num": {
        "description": "Number of devices (GPUs or TPUs) to use for training.",
        "range": "Positive integer, typically 1 or more",
    },
    "device_num": {
        "description": "Number of devices (GPUs or TPUs) to use for training.",
        "range": "Positive integer, typically 1 or more",
    },
    "learning_rate": {
        "description": "Learning rate, which controls the speed of weight updates.",
        "range": "Positive float, typically between 1e-5 and 1e-2",
    },
    "max_train_steps": {
        "description": "Maximum number of training steps. Training will stop after this many steps, regardless of the number of epochs.",
        "range": "Positive integer, typically between 10,000 and 1,000,000",
    },
    "max_seq_length": {
        "description": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
        "range": "Positive integer, typically between 128 and 512",
    },
    "batch_size": {
        "description": "Number of samples per batch during training.",
        "range": "Positive integer, typically between 1 and 64",
    },
    "val_batch_size": {
        "description": "Number of samples per batch during validation.",
        "range": "Positive integer, typically between 1 and 64",
    },
    "lr_scheduler_type": {
        "description": "Type of learning rate scheduler to use. Common options include 'linear', 'cosine', etc.",
        "range": "SchedulerType, e.g., SchedulerType.LINEAR, SchedulerType.COSINE",
    },
    "weight_decay": {
        "description": "Weight decay (L2 penalty) for the AdamW optimizer.",
        "range": "Positive float, typically between 0.01 and 0.1",
    },
    "num_warmup_steps": {
        "description": "Number of warmup steps for the learning rate scheduler. During these steps, the learning rate increases linearly from 0 to the initial learning rate.",
        "range": "Non-negative integer, typically between 0 and 10% of max_train_steps",
    },
    "gradient_accumulation_steps": {
        "description": "Number of gradient accumulation steps. This is used to simulate larger batch sizes by accumulating gradients over multiple smaller batches.",
        "range": "Positive integer, typically between 1 and 16",
    },
    "save_step_interval": {
        "description": "Interval (in steps) at which to save the model. A value of 1000 means the model will be saved every 1000 steps.",
        "range": "Positive integer, typically between 100 and 10,000",
    },
}

META_INFO = {
    "SpikeLM": {
        "model": SpikeLMConfig,
        "trainer": TrainerConfig,
        "comment": SPIKELM_CONFIG_COMMENT,
    }
}
