import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input, hidden):
        super(Encoder, self).__init__()
        self.hidden = hidden
        self.embedding = nn.Embedding(input, hidden)
        self.gru = nn.GRU(hidden, hidden)

    def forward(self, input, hidden):
        return self.gru(self.embedding(input).view(1, 1, -1), hidden)

