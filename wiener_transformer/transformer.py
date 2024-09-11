import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from wiener_transformer.utils.embeddings import create_embedding_weights, load_glove_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. 
    The Encoder processes the input sequence, and the Decoder generates the output sequence.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Linear layer followed by softmax to generate output probabilities over the target vocabulary.
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """
    Core Encoder is a stack of N identical layers.
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    Layer Normalization as introduced by Ba et al.
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer normalization.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Encoder layer is made up of self-attention and feed-forward networks.
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking to prevent attending to future positions.
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder layer is made up of self-attention, source-attention, and feed-forward networks.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention mechanism, which allows the model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.i = 0

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = scores.softmax(dim=-1)

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        
        x, self.attn = torch.matmul(p_attn, value), p_attn
    
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    """
    Embedding layer that can be initialized with pre-trained embeddings (Word2Vec, FastText, GloVe) or learned embeddings.
    """
    def __init__(self, d_model, vocab_size, embedding_type='learned', word2vec_model=None, fasttext_model=None, glove_weights=None):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.embedding_type = embedding_type
        padding_idx = 2
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        if embedding_type == 'word2vec' and word2vec_model is not None:
            self.init_word2vec_weights(word2vec_model)
            self.embedding.weight.requires_grad = False
        elif embedding_type == 'fasttext' and fasttext_model is not None:
            self.init_fasttext_weights(fasttext_model)
            self.embedding.weight.requires_grad = False
        elif embedding_type == 'glove' and glove_weights is not None:
            self.init_glove_weights(glove_weights)
            self.embedding.weight.requires_grad = False
        elif embedding_type == 'learned':
            self.init_learned_weights()
            # If learned, do not freeze the weights
        elif embedding_type == 'learned_frozen':
            self.init_learned_weights()
            # Freeze the weights for learned_frozen embeddings
            self.embedding.weight.requires_grad = False
        else:
            raise ValueError("Invalid embedding type or model not provided.")

    def init_word2vec_weights(self, word2vec_model):
        # Initialize weights from Word2Vec model
        word2vec_vocab_size = len(word2vec_model.wv)

        for i in range(self.vocab_size):
            if i < word2vec_vocab_size:
                self.embedding.weight.data[i] = torch.tensor(word2vec_model.wv.vectors[i], dtype=torch.float)
            else:
                # Random initialization for out-of-vocabulary indices
                self.embedding.weight.data[i] = torch.randn(self.d_model)

        # Scale embeddings
        self.embedding.weight.data = self.embedding.weight.data * math.sqrt(self.d_model)

    def init_fasttext_weights(self, fasttext_model):
        # Initialize weights from FastText model
        fasttext_vocab_size = len(fasttext_model.wv)

        for i in range(self.vocab_size):
            if i < fasttext_vocab_size:
                self.embedding.weight.data[i] = torch.tensor(fasttext_model.wv.vectors[i], dtype=torch.float)
            else:
                # Random initialization for out-of-vocabulary indices
                self.embedding.weight.data[i] = torch.randn(self.d_model)

        # Scale embeddings
        self.embedding.weight.data = self.embedding.weight.data * math.sqrt(self.d_model)

    def init_glove_weights(self, glove_weights):
        # Initialize weights from GloVe embeddings
        glove_vocab_size = glove_weights.size(0)

        for i in range(self.vocab_size):
            if i < glove_vocab_size:
                self.embedding.weight.data[i] = glove_weights[i]
            else:
                # Random initialization for out-of-vocabulary indices
                self.embedding.weight.data[i] = torch.randn(self.d_model)

        # Scale embeddings
        self.embedding.weight.data = self.embedding.weight.data * math.sqrt(self.d_model)

    def init_learned_weights(self):
        # Initialize with Xavier uniform for learned embeddings
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)



class PositionalEncoding(nn.Module):
    """
    Add positional encoding to the input embeddings to provide information about the position of the tokens in the sequence.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, 
    embedding_type='learned'
):
    """
    Helper function to construct a model from hyperparameters.
    
    Args:
        src_vocab: Size of source vocabulary.
        tgt_vocab: Size of target vocabulary.
        N: Number of layers in the encoder and decoder.
        d_model: Dimensionality of the embeddings.
        d_ff: Dimensionality of the feed-forward network.
        h: Number of attention heads.
        dropout: Dropout rate.
        embedding_type: Type of embeddings ('learned', 'word2vec', 'fasttext', 'glove').

    Returns:
        A constructed EncoderDecoder model.
    """
    
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    src_word2vec_model, tgt_word2vec_model = None, None
    src_fasttext_model, tgt_fasttext_model = None, None
    src_glove_weights, tgt_glove_weights = None, None

    if embedding_type == "word2vec":
        src_word2vec_model, tgt_word2vec_model = create_embedding_weights('word2vec', vector_size=d_model)
    elif embedding_type == "fasttext":
        src_fasttext_model, tgt_fasttext_model = create_embedding_weights('fasttext', vector_size=d_model)
    elif embedding_type == "glove":
        src_glove_weights, tgt_glove_weights = load_glove_embeddings(vector_size=d_model)

    src_embeddings = nn.Sequential(
        Embeddings(d_model,
                   src_vocab,
                   embedding_type,
                   src_word2vec_model,
                   src_fasttext_model,
                   src_glove_weights),
        c(position)
    )
    tgt_embeddings = nn.Sequential(
        Embeddings(d_model,
                   tgt_vocab,
                   embedding_type,
                   tgt_word2vec_model,
                   tgt_fasttext_model,
                   tgt_glove_weights),
        c(position)
    )

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embeddings,
        tgt_embeddings,
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model.to(device)
