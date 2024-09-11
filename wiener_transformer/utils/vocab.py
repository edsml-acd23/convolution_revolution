import os

from datasets import load_dataset
import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def batch_iterator(dataset, lang='en', batch_size=1000):
    """
    Generator function that yields batches of text data from the dataset.

    Args:
        dataset: The dataset to iterate over.
        lang: The language to extract text from ('en' or 'de').
        batch_size: The size of each batch to yield.

    Yields:
        Batches of text data.
    """
    for i in range(0, len(dataset), batch_size):
        yield dataset[lang][i:i + batch_size]


def transform(examples):
    """
    Transforms the dataset examples into a format suitable for tokenization.

    Args:
        examples: A dictionary containing 'translation' key with text data.

    Returns:
        A dictionary with separate keys for source ('en') and target ('de') languages.
    """
    return {'de': [t['de'] for t in examples['translation']],
            'en': [t['en'] for t in examples['translation']]}


def load_wmt(language_pair=('en', 'de')):
    """
    Loads the WMT dataset for the specified language pair using the HuggingFace datasets library.

    Args:
        language_pair: A tuple specifying the source and target languages (default is ('en', 'de')).

    Returns:
        The training, validation, and test splits of the WMT dataset.
    """
    language_pair_str = f"{language_pair[1]}-{language_pair[0]}"
    train = load_dataset("wmt/wmt14", language_pair_str, split="train")
    val = load_dataset("wmt/wmt14", language_pair_str, split="validation")
    test = load_dataset("wmt/wmt14", language_pair_str, split="test")
    print("Loaded WMT datasets")

    return train, val, test


def load_tokenizers(dataset):
    """
    Loads or trains tokenizers for the source and target languages. If pre-trained tokenizers are available on disk,
    they are loaded; otherwise, new tokenizers are trained on the dataset.

    Args:
        dataset: The dataset to use for training the tokenizers if needed.

    Returns:
        The source and target tokenizers.
    """
    prefix = os.environ.get("base_dir", "/scratch_brain/acd23/code/irp-acd23")
    if os.path.exists(prefix + "/tokenizers/src_tokenizer.json") and os.path.exists(prefix + "/tokenizers/tgt_tokenizer.json"):
        tokenizer_src = Tokenizer.from_file(prefix + "/tokenizers/src_tokenizer.json")
        tokenizer_tgt = Tokenizer.from_file(prefix + "/tokenizers/tgt_tokenizer.json")
        print("Loaded tokenizers from file")
    else:
        tokenizer_src = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer_tgt = Tokenizer(BPE(unk_token="<unk>"))
        
        trainer_src = BpeTrainer(special_tokens=["<s>", "</s>", "<blank>", "<unk>", "</w>"])
        trainer_tgt = BpeTrainer(special_tokens=["<s>", "</s>", "<blank>", "<unk>", "</w>"])
        
        dataset = dataset.map(transform, batched=True, remove_columns=['translation'])
        dataset_list = dataset[:]
        tokenizer_src.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()
        tokenizer_tgt.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()
        tokenizer_src.post_processor = tokenizers.processors.ByteLevel()
        tokenizer_tgt.post_processor = tokenizers.processors.ByteLevel()
        tokenizer_src.decoder = tokenizers.decoders.ByteLevel()
        tokenizer_tgt.decoder = tokenizers.decoders.ByteLevel()
        tokenizer_src.train_from_iterator(batch_iterator(dataset_list, "en"), trainer = trainer_src, length = len(dataset_list))
        tokenizer_tgt.train_from_iterator(batch_iterator(dataset_list, "de"), trainer = trainer_tgt, length = len(dataset_list))
        
        print("Trained tokenizers")

        tokenizer_src.save(prefix + "/tokenizers/src_tokenizer.json")
        tokenizer_tgt.save(prefix + "/tokenizers/tgt_tokenizer.json")

        print("Saved tokenizers to file")

    return tokenizer_src, tokenizer_tgt
