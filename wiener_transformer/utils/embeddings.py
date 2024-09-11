import os

import torch
import numpy as np
from gensim.models import Word2Vec, FastText
from tqdm import tqdm

from wiener_transformer.utils.data_loader import create_dataloaders
from wiener_transformer.utils.vocab import load_wmt, load_tokenizers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_glove_embeddings(vector_size=512):
    base_dir = os.environ.get("base_dir", "/scratch_brain/acd23/code/irp-acd23")
    
    src_vocab_path = os.path.join(base_dir, f"embeddings/src_glove_vocab_{vector_size}.txt")
    tgt_vocab_path = os.path.join(base_dir, f"embeddings/tgt_glove_vocab_{vector_size}.txt")
    src_vectors_path = os.path.join(base_dir, f"embeddings/src_glove_vectors_{vector_size}.txt")
    tgt_vectors_path = os.path.join(base_dir, f"embeddings/tgt_glove_vectors_{vector_size}.txt")

    def load_vectors_and_vocab(vectors_path, vocab_path):
        embeddings_index = {}
        with open(vectors_path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        with open(vocab_path, 'r') as f:
            vocab = [line.split()[0] for line in f.readlines()]

        embedding_matrix = np.zeros((len(vocab), vector_size))
        for i, word in enumerate(vocab):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.random.normal(scale=0.6, size=(vector_size,))
        
        return torch.tensor(embedding_matrix, dtype=torch.float)

    # Load source and target embeddings
    src_embedding_matrix = load_vectors_and_vocab(src_vectors_path, src_vocab_path)
    tgt_embedding_matrix = load_vectors_and_vocab(tgt_vectors_path, tgt_vocab_path)
    
    return src_embedding_matrix, tgt_embedding_matrix


def create_embedding_weights(model_type='word2vec', vector_size=512, batch_size=1):
    file_dir = os.path.join(os.environ.get("base_dir", "/scratch_brain/acd23/code/irp-acd23"), "embeddings")
    src_filename = os.path.join(file_dir, f"src_{model_type}_{vector_size}.model")
    tgt_filename = os.path.join(file_dir, f"tgt_{model_type}_{vector_size}.model")

    src_embed_exists = os.path.exists(src_filename)
    tgt_embed_exists = os.path.exists(tgt_filename)

    if src_embed_exists and tgt_embed_exists:
        if model_type == 'word2vec':
            src_embed = Word2Vec.load(src_filename)
            tgt_embed = Word2Vec.load(tgt_filename)
        elif model_type == 'fasttext':
            src_embed = FastText.load(src_filename)
            tgt_embed = FastText.load(tgt_filename)
        print(f"Loaded existing {model_type} embeddings for vector size {vector_size}.")
    else:
        train, val, test = load_wmt()
        src_tokenizer, tgt_tokenizer = load_tokenizers(train)

        train_loader, _, _ = create_dataloaders(
                train,
                val,
                test,
                src_tokenizer,
                tgt_tokenizer,
                device,
                batch_size=batch_size,
                max_padding=200
            )
        
        print("Loaded data for training embeddings.")

        src_sentences, tgt_sentences = extract_sentences(train_loader)
        print("Extracted sentences for training embeddings.")

        if model_type == 'word2vec':
            src_embed = Word2Vec(sentences=src_sentences, vector_size=vector_size)
            tgt_embed = Word2Vec(sentences=tgt_sentences, vector_size=vector_size)
        elif model_type == 'fasttext':
            src_embed = FastText(sentences=src_sentences, vector_size=vector_size)
            tgt_embed = FastText(sentences=tgt_sentences, vector_size=vector_size)

        os.makedirs(file_dir, exist_ok=True)

        src_embed.save(src_filename)
        tgt_embed.save(tgt_filename)
        print(f"Trained and saved new {model_type} embeddings for vector size {vector_size}.")
        del train_loader
        del train, val, test
        del src_tokenizer, tgt_tokenizer
        del src_sentences, tgt_sentences

    return src_embed, tgt_embed

def extract_sentences(dataloader):
    src_sentences = []
    tgt_sentences = []
    for batch in tqdm(dataloader, desc="Extracting sentences"):
        src_sentences.extend(batch[0].tolist())
        tgt_sentences.extend(batch[1].tolist())
    return src_sentences, tgt_sentences