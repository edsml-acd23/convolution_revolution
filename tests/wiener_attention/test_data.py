import pytest
import torch
from transformers import BertTokenizer

from wiener_attention.data import (
    load_imdb,
    collate_batch,
    IMDBDataset,
    create_dataloaders
)


@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')


def test_load_imdb():
    train, val, test = load_imdb()
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert 'text' in train[0]
    assert 'label' in train[0]


def test_collate_batch():
    device = torch.device("cpu")
    batch = [
        {"text": "This is a positive review.", "label": 1},
        {"text": "This is a negative review.", "label": 0}
    ]

    def mock_text_pipeline(text):
        return [1, 2, 3, 4, 5], [1, 1, 1, 1, 1]

    text_tensor, label_tensor, attention_mask_tensor = collate_batch(
        batch, mock_text_pipeline, device, max_padding=10
    )

    assert text_tensor.shape == (2, 10)
    assert label_tensor.shape == (2,)
    assert attention_mask_tensor.shape == (2, 10)


def test_imdb_dataset():
    data = [
        {"text": "Positive review", "label": 1},
        {"text": "Negative review", "label": 0}
    ]
    dataset = IMDBDataset(data)

    assert len(dataset) == 2
    assert dataset[0] == data[0]
    assert dataset[1] == data[1]


def test_create_dataloaders(tokenizer):
    train_data = [{"text": "Train text", "label": 1}] * 10
    valid_data = [{"text": "Valid text", "label": 0}] * 5
    test_data = [{"text": "Test text", "label": 1}] * 5

    device = torch.device("cpu")

    train_loader, valid_loader, test_loader = create_dataloaders(
        train_data, valid_data, test_data, tokenizer, device, batch_size=2
    )

    assert len(train_loader) == 5  # 10 samples / batch_size 2
    assert len(valid_loader) == 3  # 5 samples / batch_size 2 (rounded up)
    assert len(test_loader) == 3   # 5 samples / batch_size 2 (rounded up)

    batch = next(iter(train_loader))
    assert len(batch) == 3  # input_ids, labels, attention_mask
    assert batch[0].shape == (2, 512)  # batch_size, max_padding
    assert batch[1].shape == (2,)      # batch_size
    assert batch[2].shape == (2, 512)  # batch_size, max_padding
