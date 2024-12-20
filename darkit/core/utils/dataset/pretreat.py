import random
import numpy as np


def mask_single(example, tokenizer):
    labels = np.copy(example)
    probability_matrix = np.full(np.array(example).shape, 0.15)
    special_tokens_mask = tokenizer.get_special_tokens_mask(
        example, already_has_special_tokens=True
    )
    special_tokens_mask = np.array(special_tokens_mask, dtype=np.bool_)
    probability_matrix[special_tokens_mask] = 0

    masked_indices = np.random.binomial(
        1, probability_matrix, size=probability_matrix.shape
    ).astype(np.bool_)
    # print(labels, masked_indices)
    labels[~masked_indices] = -100

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        np.random.binomial(1, 0.8, size=labels.shape).astype(np.bool_) & masked_indices
    )
    example_out = np.array(example)
    example_out[indices_replaced] = tokenizer.mask_token_id

    indices_random = (
        np.random.binomial(1, 0.5, size=labels.shape).astype(np.bool_)
        & masked_indices
        & ~indices_replaced
    )
    random_words = np.random.randint(
        low=0,
        high=len(tokenizer),
        size=np.count_nonzero(indices_random),
        dtype=np.int64,
    )
    example_out[indices_random] = random_words
    return example_out, labels


def create_next_sentence(sentences):
    """
    next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
        pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

        - 0 indicates sequence B is a continuation of sequence A,
        - 1 indicates sequence B is a random sequence.
    """
    i = 0
    new_sentences = []
    next_sentence_labels = []
    while True:
        s1 = sentences[i]
        next_sentence_label = random.choice([0, 1])
        i += 1
        if next_sentence_label == 0:
            s2 = sentences[i]
            i += 1
        else:
            j = list(range(0, i - 1)) + list(range(i + 1, len(sentences)))
            s2 = sentences[random.choice(j)]

        new_sentences.append(s1 + s2)
        next_sentence_labels.append(next_sentence_label)
        if i >= len(sentences) - 2:
            break

    return new_sentences, next_sentence_labels
