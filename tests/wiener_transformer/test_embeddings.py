import os

import torch

from wiener_transformer.utils.embeddings import (
    load_glove_embeddings, create_embedding_weights)


def test_load_glove_embeddings():

    try:
        src_embedding_matrix, tgt_embedding_matrix = load_glove_embeddings(vector_size=512)
        assert isinstance(src_embedding_matrix, torch.Tensor), "Source embedding matrix is not a tensor."
        assert isinstance(tgt_embedding_matrix, torch.Tensor), "Target embedding matrix is not a tensor."
        assert src_embedding_matrix.shape[1] == 512, "Source embedding matrix has incorrect vector size."
        assert tgt_embedding_matrix.shape[1] == 512, "Target embedding matrix has incorrect vector size."
        print("test_load_glove_embeddings passed.")
    except Exception as e:
        print(f"test_load_glove_embeddings failed: {e}")


def test_create_embedding_weights():

    try:
        src_embed, tgt_embed = create_embedding_weights(model_type='word2vec', vector_size=512, batch_size=1)
        assert src_embed.vector_size == 100, "Source Word2Vec model has incorrect vector size."
        assert tgt_embed.vector_size == 100, "Target Word2Vec model has incorrect vector size."
        print("test_create_embedding_weights passed for Word2Vec.")

        src_embed, tgt_embed = create_embedding_weights(model_type='fasttext', vector_size=512, batch_size=1)
        assert src_embed.vector_size == 100, "Source FastText model has incorrect vector size."
        assert tgt_embed.vector_size == 100, "Target FastText model has incorrect vector size."
        print("test_create_embedding_weights passed for FastText.")
    except Exception as e:
        print(f"test_create_embedding_weights failed: {e}")


if __name__ == "__main__":
    os.environ["base_dir"] = "."

    test_load_glove_embeddings()
    test_create_embedding_weights()
