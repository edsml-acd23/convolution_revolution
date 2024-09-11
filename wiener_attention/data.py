import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


def pad(tensor, pad, value):
    return torch.nn.functional.pad(tensor, pad, value = value)


def load_imdb():
    """
    Load the IMDB dataset for sentiment analysis.

    This function loads the IMDB dataset, splitting it into train, validation, and test sets.
    The train and validation sets are created by splitting the original training data,
    while the test set is loaded separately.

    Returns:
        tuple: A tuple containing three Dataset objects:
            - train (Dataset): The training dataset (90% of the original training data)
            - val (Dataset): The validation dataset (10% of the original training data)
            - test (Dataset): The test dataset

    Note:
        This function uses the Hugging Face datasets library to load the IMDB dataset.
    """
    split = load_dataset("stanfordnlp/imdb", split="train").train_test_split(test_size=0.1)
    train = split['train']
    val = split['test']
    test = load_dataset("stanfordnlp/imdb", split="test")
    print("Loaded IMDB datasets")

    return train, val, test


def collate_batch(
    batch,
    text_pipeline,
    device,
    max_padding=128,
    pad_id=0,
    attention_mask=True
):
    """
    Collate a batch of examples into a single tensor.

    Args:
        batch (list): A list of examples from the dataset.
        text_pipeline (callable): A function to process the text into tokens and attention mask.
        device (torch.device): The device to put the tensors on.
        max_padding (int, optional): Maximum length to pad sequences to. Defaults to 128.
        pad_id (int, optional): The ID to use for padding. Defaults to 0.
        attention_mask (bool, optional): Whether to return attention masks. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - text_tensor (torch.Tensor): Padded and stacked input IDs.
            - label_tensor (torch.Tensor): Stacked labels.
            - attention_mask_tensor (torch.Tensor, optional): Padded and stacked attention masks.
    """
    text_list, attention_mask_list, label_list = [], [], []
    
    for example in batch:
        _text = example["text"]
        _label = example["label"]
        
        tokenized_text, attention_mask = text_pipeline(_text)
        # Process the text using the text pipeline
        processed_text = torch.cat(
            [
                torch.tensor(
                    tokenized_text,
                    dtype=torch.int64,
                    device=device,
                ),
            ],
            0,
        )
        
        # Append padded text to the text list
        text_list.append(
            pad(
                processed_text,
                (0, max_padding - len(processed_text)),
                value=pad_id,
            )
        )

        # Append attention mask to the attention mask list
        attention_mask_list.append(
            pad(
                torch.tensor(
                    attention_mask,
                    dtype=torch.int64,
                    device=device,
                ),
                (0, max_padding - len(attention_mask)),
                value=0,
            )
        )
        
        label_list.append(torch.tensor(_label, dtype=torch.int64, device=device))

    text_tensor = torch.stack(text_list)
    label_tensor = torch.stack(label_list)
    
    if attention_mask:
        attention_mask_tensor = torch.stack(attention_mask_list)
        return (text_tensor, label_tensor, attention_mask_tensor)
    
    return (text_tensor, label_tensor)


class IMDBDataset(Dataset):
    """
    A custom Dataset class for the IMDB dataset.

    This class wraps the IMDB dataset and provides methods required
    for use with PyTorch DataLoader.

    Attributes:
        data (list): The list of data samples.
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
    tokenizer,
    device,
    batch_size=32,
    max_padding=512,
    attention_mask=True
):
    """
    Create DataLoader objects for training, validation, and testing.

    Args:
        train_data (Dataset): The training dataset.
        valid_data (Dataset): The validation dataset.
        test_data (Dataset): The test dataset.
        tokenizer: The tokenizer to use for processing text.
        device (torch.device): The device to put the tensors on.
        batch_size (int, optional): The batch size for the DataLoaders. Defaults to 32.
        max_padding (int, optional): Maximum length to pad sequences to. Defaults to 512.
        attention_mask (bool, optional): Whether to include attention masks. Defaults to True.

    Returns:
        tuple: A tuple containing three DataLoader objects:
            - train_dataloader (DataLoader): DataLoader for the training data.
            - valid_dataloader (DataLoader): DataLoader for the validation data.
            - test_dataloader (DataLoader): DataLoader for the test data.
    """
    vocab = tokenizer.get_vocab()

    def tokenize(text):
        output = tokenizer(text, truncation=True, max_length=512)
        return output["input_ids"], output["attention_mask"]

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize,
            device,
            max_padding=max_padding,
            pad_id=vocab["[PAD]"],
            attention_mask=attention_mask
        )

    train_dataset = IMDBDataset(train_data)
    valid_dataset = IMDBDataset(valid_data)
    test_dataset = IMDBDataset(test_data)

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



