# SpikeGPT

## GPTConfig
* vocab_size: 词汇表大小。表示可以处理不同token的数量。词汇表大小直接影响模型的参数数量，较大的词汇表意味着嵌入层和输出层的参数会更多。
* ctx_len：上下文长度。模型在处理输入序列时所能考虑的最大序列长度。较长的上下文长度需要更多的计算资源。**Increase T_MAX in model.py if your ctx_len > 1024**
* model_type: 指定架构类型，有RWKV和RWKV-ffnPre两种供选择。'RWKV' (better for char-level English) or 'RWKV-ffnPre' (better in some cases)
* n_layer: Block层数。Block架构见论文Figure1或model.py的Block类。
* n_embd: 嵌入维度。决定了模型中嵌入层(embedding layer)的向量维度。
