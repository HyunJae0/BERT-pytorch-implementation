import torch
import torch.nn as nn

class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.hidden_size = config.hidden_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        self.segment_embedding = nn.Embedding(3, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)
        embedding = self.token_embedding(input_ids) + self.segment_embedding(segment_ids) + self.position_embedding(position_ids)
        embedding = self.dropout(self.layer_norm(embedding))
        return embedding
