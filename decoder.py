import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, hidden, output):
        super(Decoder, self).__init__()
        self.hidden = hidden

        self.embedding = nn.Embedding(output, hidden)
        self.gru = nn.GRU(hidden, hidden)
        self.output = nn.Linear(hidden, output)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.gru(torch.nn.functional.relu(self.embedding(input).view(1, 1, -1)), hidden)
        return self.softmax(self.output(output[0])), hidden
