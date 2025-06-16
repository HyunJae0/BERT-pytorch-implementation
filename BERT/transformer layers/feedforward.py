import torch.nn as nn

class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_dropout_prob = config.hidden_dropout_prob

        self.linear1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.linear2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.activation = nn.GELU()

    def forward(self, hidden_state):
        hidden_state = self.linear2(self.dropout(self.activation(self.linear1(hidden_state))))
        return hidden_state
