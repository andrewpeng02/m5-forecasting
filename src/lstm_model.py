from torch import nn


class M5LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    # seqs shape [batch_size, timesteps, features]
    # out shape [batch_size * timesteps, 1]
    def forward(self, seqs, hidden):
        out, hidden = self.lstm(seqs, hidden)
        out = out.reshape(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
                 weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())

        return hidden

