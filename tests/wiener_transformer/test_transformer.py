import pytest
import torch
import torch.nn as nn
from wiener_transformer.transformer import (
    Generator, LayerNorm, SublayerConnection, MultiHeadedAttention,
    PositionwiseFeedForward, Embeddings,
    PositionalEncoding, subsequent_mask, clones, make_model
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def src_vocab():
    return 10

@pytest.fixture
def tgt_vocab():
    return 10

@pytest.fixture
def transformer_model(src_vocab, tgt_vocab):
    return make_model(src_vocab, tgt_vocab)


def test_encoder_decoder_forward(transformer_model, device):
    src = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device=device)
    tgt = torch.tensor([[1, 2, 3], [3, 2, 1]], device=device)
    src_mask = torch.ones(2, 1, 4, device=device)
    tgt_mask = subsequent_mask(tgt.size(1)).to(device)
    out = transformer_model(src, tgt, src_mask, tgt_mask)
    logits = transformer_model.generator(out)
    assert logits.shape == (2, 3, transformer_model.generator.proj.out_features)


def test_generator_forward(device):
    vocab_size = 10
    d_model = 512
    generator = Generator(d_model, vocab_size).to(device)
    x = torch.randn(1, 5, d_model, device=device)
    out = generator(x)
    probs = torch.exp(out)
    assert out.shape == (1, 5, vocab_size)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor(1.0, device=device))


def test_layer_norm_forward():
    features = 512
    ln = LayerNorm(features)
    x = torch.randn(10, 512)
    out = ln(x)
    assert out.shape == x.shape


def test_sublayer_connection_forward():
    size = 512
    dropout = 0.1
    sublayer_connection = SublayerConnection(size, dropout)
    x = torch.randn(10, 512)
    sublayer = nn.Linear(size, size)
    out = sublayer_connection(x, sublayer)
    assert out.shape == x.shape


def test_multi_headed_attention(device):
    h = 8
    d_model = 512
    mha = MultiHeadedAttention(h, d_model).to(device)
    x = torch.randn(10, 5, d_model, device=device)
    mask = torch.ones(10, 5, 5, device=device)
    out = mha(x, x, x, mask)
    assert out.shape == x.shape


def test_positionwise_feedforward(device):
    d_model = 512
    d_ff = 2048
    ff = PositionwiseFeedForward(d_model, d_ff).to(device)
    x = torch.randn(10, 5, d_model, device=device)
    out = ff(x)
    assert out.shape == x.shape


def test_embeddings(device, src_vocab):
    d_model = 512
    embeddings = Embeddings(d_model, src_vocab).to(device)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
    out = embeddings(x)
    assert out.shape == (2, 3, d_model)


def test_positional_encoding(device):
    d_model = 512
    dropout = 0.1
    max_len = 60
    pe = PositionalEncoding(d_model, dropout, max_len).to(device)
    x = torch.randn(10, 20, d_model, device=device)
    out = pe(x)
    assert out.shape == x.shape


def test_subsequent_mask():
    size = 5
    mask = subsequent_mask(size)
    expected_mask = torch.tril(torch.ones(size, size)).unsqueeze(0).bool()
    assert mask.shape == (1, size, size)
    assert torch.equal(mask, expected_mask)


def test_clones():
    size = 512
    dropout = 0.1
    sublayer_connection = SublayerConnection(size, dropout)
    N = 6
    clones_list = clones(sublayer_connection, N)
    assert len(clones_list) == N
    for layer in clones_list:
        assert isinstance(layer, SublayerConnection)
