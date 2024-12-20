import json
from darkit.cli.src.options import get_option_definer
from ..utils import TMP_PATH


OPTION_TMP_PATH = TMP_PATH / "options"


def get_models_options(key):
    """
    generate options file for all models.
    """
    KEY_OPTION_PATH = OPTION_TMP_PATH / f"{key}.json"
    with open(KEY_OPTION_PATH, "r") as f:
        return json.load(f)


def save_models_options(key, options):
    """
    save options file for all models.
    """
    if not OPTION_TMP_PATH.exists():
        OPTION_TMP_PATH.mkdir(parents=True)
    KEY_OPTION_PATH = OPTION_TMP_PATH / f"{key}.json"
    with open(KEY_OPTION_PATH, "w") as f:
        options = json.dumps(options, indent=4)
        f.write(options)


def save_models_metadata(key, metadata):
    """
    save metadata file for all models.
    """
    if not OPTION_TMP_PATH.exists():
        OPTION_TMP_PATH.mkdir(parents=True)
    option_definer = dict()
    for model in metadata:
        value = metadata[model]
        mconf = value.get("model")
        tconf = value.get("trainer")
        conf_comment = value.get("comment", DEFAULT_CONFIG_COMMENT)
        model_option_definer = (
            get_option_definer(mconf, conf_comment) if mconf else None
        )
        trainer_option_definer = (
            get_option_definer(tconf, conf_comment) if tconf else None
        )

        option_definer[model] = {
            "model": model_option_definer,
            "trainer": trainer_option_definer,
        }
    KEY_OPTION_PATH = OPTION_TMP_PATH / f"{key}.json"
    with open(KEY_OPTION_PATH, "w") as f:
        config_json = json.dumps(option_definer, indent=4)
        f.write(config_json)


DEFAULT_CONFIG_COMMENT = {
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
