from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer
)


def make_bert_model(
    wiener_attention=None,
    wiener_similarity=None,
    num_labels=2,
    gamma=None
):
    """
    Initializes a single-layer BERT model for sequence classification with a
    custom attention module.

    Args:
        num_labels (int): Number of labels for the classification task.

    Returns:
        tokenizer: The tokenizer associated with the model.
        model: A BERT model instance for sequence classification with
            custom attention.
    """
    config = BertConfig(
        vocab_size=30522,  # Number of tokens in the vocabulary
        hidden_size=64,   # Hidden size
        num_hidden_layers=1,  # Number of transformer layers
        num_attention_heads=1,  # Number of attention heads
        intermediate_size=3072,  # Intermediate size in feed-forward layer
        num_labels=num_labels,  # Number of labels for the classification task
    )

    # Initialize the BERT model with the created configuration
    model = BertForSequenceClassification(config)

    if wiener_attention is not None:
        for param in wiener_similarity.parameters():
            param.requires_grad = False
       
        model.bert.encoder.layer[0].attention.self = wiener_attention(
            config, wiener_similarity, gamma=gamma
        )

    return model


def make_bert_tokenizer():
    """
    Initializes a BERT tokenizer.

    Returns:
        tokenizer: The tokenizer associated with the model.
    """
    # Initialize a BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return tokenizer
