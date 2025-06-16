import torch.nn as nn
from transformer_layers.attention import MultiHeadAttention
from transformer_layers.feedforward import FeedForwardLayer

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForwardLayer(config)

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_state, mask):
        ## Post-Norm
        _hidden_state = hidden_state
        attn_output = self.attention(hidden_state, hidden_state, hidden_state, mask)  # attention
        hidden_state = _hidden_state + attn_output  # residual
        norm_x1 = self.norm1(hidden_state)  # layernorm
        hidden_state = self.dropout1(norm_x1)

        _hidden_state = hidden_state
        ffn_output = self.ffn(hidden_state)  # feed-forward network
        hidden_state = _hidden_state + ffn_output  # residual
        norm_x2 = self.norm2(hidden_state)  # layernorm
        hidden_state = self.dropout2(norm_x2)
        return hidden_state
