import torch
from torch.utils.data import DataLoader, Dataset

from wiener_transformer.utils.helpers import pad


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    device,
    max_padding=128,
    pad_id=2,
):
    """
    Collate a batch of data for the DataLoader, applying tokenization, padding, and conversion to tensors.

    Args:
        batch: A list of data examples where each example is a dictionary containing 'translation' with 'en' and 'de' keys.
        src_pipeline: A tokenization function for the source text.
        tgt_pipeline: A tokenization function for the target text.
        device: The device to place the tensors on (e.g., 'cpu' or 'cuda').
        max_padding: The maximum length to pad the sequences to (default: 128).
        pad_id: The token ID used for padding (default: 2).

    Returns:
        A tuple of tensors (src, tgt) where:
            - src: Padded and tokenized source sequences.
            - tgt: Padded and tokenized target sequences.
    """
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for example in batch:
        _src = example["translation"]["en"]
        _tgt = example["translation"]["de"]
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_pipeline(_src),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_pipeline(_tgt),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


class WMT14Dataset(Dataset):
    """
    A custom Dataset class for the WMT14 dataset.

    Args:
        data_iter: An iterable containing the data samples.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the data sample at the given index.
    """
    def __init__(self, data_iter):
        self.data = list(data_iter)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloaders(
    train_data,
    valid_data,
    test_data,
    src_tokenizer,
    tgt_tokenizer,
    device,
    batch_size=32,
    max_padding=128
):
    """
    Create DataLoader objects for the training, validation, and test datasets.

    Args:
        train_data: Iterable containing the training data.
        valid_data: Iterable containing the validation data.
        test_data: Iterable containing the test data.
        src_tokenizer: Tokenizer for the source language.
        tgt_tokenizer: Tokenizer for the target language.
        device: The device to place the tensors on (e.g., 'cpu' or 'cuda').
        batch_size: Number of samples per batch (default: 32).
        max_padding: The maximum length to pad the sequences to (default: 128).

    Returns:
        Tuple containing DataLoader objects for the training, validation, and test datasets.
    """
    src_vocab = src_tokenizer.get_vocab()

    def tokenize_src(text):
        return src_tokenizer.encode(text).ids
    
    def tokenize_tgt(text):
        return tgt_tokenizer.encode(text).ids

    def collate_fn(batch):
        return collate_batch(
                batch,
                tokenize_src,
                tokenize_tgt,
                device,
                max_padding=max_padding,
                pad_id=src_vocab["<blank>"],
            )

    train_dataset = WMT14Dataset(train_data)
    valid_dataset = WMT14Dataset(valid_data)
    test_dataset = WMT14Dataset(test_data)

    train_sampler = None
    valid_sampler = None
    test_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    print("Train dataloader created")

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )

    print("Validation dataloader created")
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=(test_sampler is None),
        sampler=test_sampler,
        collate_fn=collate_fn,
    )

    print("Test dataloader created")
    return train_dataloader, valid_dataloader, test_dataloader