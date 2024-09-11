import pytest

from transformers import (
    BertForSequenceClassification,
    BertTokenizer
)

from wiener_attention.model import make_bert_model, make_bert_tokenizer
from wiener_attention.attention_mechanism import WienerSelfAttention
from wiener_attention.wiener_metric import WienerSimilarityMetric


def test_make_bert_model_default():
    model = make_bert_model()
    assert isinstance(model, BertForSequenceClassification)
    assert model.config.num_labels == 2
    assert model.config.num_hidden_layers == 1
    assert model.config.num_attention_heads == 1


def test_make_bert_model_custom_labels():
    model = make_bert_model(num_labels=3)
    assert model.config.num_labels == 3


def test_make_bert_model_with_wiener_attention():
    wiener_similarity = WienerSimilarityMetric(filter_dim=1, epsilon=1e-5, rel_epsilon=True)
    model = make_bert_model(
        wiener_attention=WienerSelfAttention,
        wiener_similarity=wiener_similarity,
        gamma=0.1
    )
    assert isinstance(model.bert.encoder.layer[0].attention.self, WienerSelfAttention)


def test_make_bert_tokenizer():
    tokenizer = make_bert_tokenizer()
    assert isinstance(tokenizer, BertTokenizer)
    assert tokenizer.vocab_size == 30522


def test_model_output_shape():
    model = make_bert_model()
    tokenizer = make_bert_tokenizer()

    input_text = "This is a test sentence."
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)

    assert outputs.logits.shape == (1, 2)  # (batch_size, num_labels)


@pytest.mark.parametrize("num_labels", [2, 3, 5])
def test_model_output_shape_different_labels(num_labels):
    model = make_bert_model(num_labels=num_labels)
    tokenizer = make_bert_tokenizer()

    input_text = "This is another test sentence."
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)

    assert outputs.logits.shape == (1, num_labels)
