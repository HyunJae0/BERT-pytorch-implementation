import torch.nn as nn
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_probs_dropout = nn.Dropout(config.attention_probs_dropout_prob) # 0.1
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(2, 3) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = self.attention_probs_dropout(self.softmax(scores))
        return weights @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attention = ScaleDotProductAttention(config)

        self.w_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_concat = nn.Linear(config.hidden_size, config.hidden_size)


    def forward(self, q, k, v, mask=None):
        B, T, H = q.shape[0], q.shape[1], q.shape[-1]

        Q = self.w_q(q).view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)
        K = self.w_k(k).view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)
        V = self.w_v(v).view(B, T, self.num_heads, H // self.num_heads).transpose(1, 2)

        output = self.attention(Q, K, V, mask=mask)
        B, N, T, H = output.shape
        output = output.transpose(1, 2).contiguous().view(B, T, N*H)
        output = self.w_concat(output)
        return output