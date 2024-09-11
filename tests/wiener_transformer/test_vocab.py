import os

from datasets import Dataset
from tokenizers import Tokenizer

from wiener_transformer.utils.vocab import (
    load_wmt, load_tokenizers,
    transform, batch_iterator
)


def test_load_wmt():
    try:
        train, val, test = load_wmt(language_pair=('en', 'de'))
        assert isinstance(train, Dataset), "Train set is not a Dataset object."
        assert isinstance(val, Dataset), "Validation set is not a Dataset object."
        assert isinstance(test, Dataset), "Test set is not a Dataset object."
        print("test_load_wmt passed.")
    except Exception as e:
        print(f"test_load_wmt failed: {e}")


def test_transform():
    sample_data = {
        'translation': [
            {'de': 'Hallo Welt', 'en': 'Hello World'},
            {'de': 'Guten Morgen', 'en': 'Good Morning'}
        ]
    }
    try:
        transformed = transform(sample_data)
        assert 'de' in transformed and 'en' in transformed, "Transformed data missing 'de' or 'en' keys."
        assert transformed['de'] == ['Hallo Welt', 'Guten Morgen'], "German translation incorrect."
        assert transformed['en'] == ['Hello World', 'Good Morning'], "English translation incorrect."
        print("test_transform passed.")
    except Exception as e:
        print(f"test_transform failed: {e}")


def test_load_tokenizers():
    try:
        train, val, test = load_wmt(language_pair=('en', 'de'))
        tokenizer_src, tokenizer_tgt = load_tokenizers(train)
        assert isinstance(tokenizer_src, Tokenizer), "Source tokenizer is not a Tokenizer object."
        assert isinstance(tokenizer_tgt, Tokenizer), "Target tokenizer is not a Tokenizer object."
        print("test_load_tokenizers passed.")
    except Exception as e:
        print(f"test_load_tokenizers failed: {e}")


def test_batch_iterator():
    sample_data = {
        'en': ['Hello World', 'Good Morning', 'Good Night', 'See you', 'Take care'],
        'de': ['Hallo Welt', 'Guten Morgen', 'Gute Nacht', 'Bis bald', 'Pass auf']
    }
    dataset = Dataset.from_dict(sample_data)
    
    try:
        iterator = batch_iterator(dataset, lang='en', batch_size=2)
        batches = list(iterator)
        assert len(batches) == 3, "Incorrect number of batches."
        assert batches[0] == ['Hello World', 'Good Morning'], "First batch content incorrect."
        print("test_batch_iterator passed.")
    except Exception as e:
        print(f"test_batch_iterator failed: {e}")


if __name__ == "__main__":
    os.environ["base_dir"] = "."
    
    test_load_wmt()
    test_transform()
    test_load_tokenizers()
    test_batch_iterator()
