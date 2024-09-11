import math

import torch
from torch import nn
import torch.nn.functional as F


class WienerSelfAttention(nn.Module):
    """
    Wiener Self-Attention mechanism for transformer models.

    This class implements a custom self-attention mechanism using Wiener filters
    for similarity computation between query and key vectors.

    Args:
        config (object): Configuration object containing model parameters.
        similarity_function (callable): Function to compute similarity between query and key vectors.
        gamma (float, optional): Parameter for the similarity function. Defaults to 0.1.

    Attributes:
        num_attention_heads (int): Number of attention heads.
        attention_head_size (int): Size of each attention head.
        all_head_size (int): Total size of all attention heads.
        query (nn.Linear): Linear layer for query transformation.
        key (nn.Linear): Linear layer for key transformation.
        value (nn.Linear): Linear layer for value transformation.
        dropout (nn.Dropout): Dropout layer for regularization.
        similarity_function (callable): Function to compute similarity.
        gamma (float): Parameter for the similarity function.
    """

    def __init__(self, config, similarity_function, gamma=0.1):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.similarity_function = similarity_function.forward
        self.gamma = gamma

    def transpose_for_scores(self, x):
        """
        Transpose and reshape the input tensor for attention score calculation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, all_head_size).

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, num_attention_heads, seq_length, attention_head_size).
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=True):
        """
        Forward pass of the Wiener Self-Attention mechanism.

        Args:
            hidden_states (torch.Tensor): Input hidden states.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            head_mask (torch.Tensor, optional): Mask for attention heads. Defaults to None.
            encoder_hidden_states (torch.Tensor, optional): Hidden states from encoder. Defaults to None.
            encoder_attention_mask (torch.Tensor, optional): Attention mask for encoder. Defaults to None.
            past_key_value (tuple, optional): Cached key and value projection states. Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to True.

        Returns:
            tuple: A tuple containing:
                - context_layer (torch.Tensor): Output context layer.
                - attention_probs (torch.Tensor): Attention probabilities if output_attentions is True.
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        batch_size, num_heads, num_queries, d_k = query_layer.size()
        num_keys = key_layer.size(2)

        k = key_layer.unsqueeze(3).expand(-1, -1, -1, num_queries, -1).reshape(-1, d_k)
        q = query_layer.repeat(1, 1, num_keys, 1).reshape(-1, d_k)

        attention_scores = self.similarity_function(q, k,self.gamma).reshape(batch_size, num_heads, num_queries, num_keys)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_mask = attention_mask * -1
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmin(attention_scores, dim=-1)
        self.attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            self.attention_probs = self.attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, self.attention_probs) if output_attentions else (context_layer,)
        
        return outputs