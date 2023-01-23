import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, h_enc, h_dec):
        '''
        h_enc: (batch_size, max_len, hidden_dim)
        h_dec: (batch_size, hidden_dim)
        '''
        # print("in attention")
        # h_dec = h_dec.unsqueeze(1) # (batch_size, 1, hidden_dim)
        h_enc = self.W1(h_enc) # (batch_size, max_len, hidden_dim)
        h_dec = self.W2(h_dec) # (batch_size, 1, hidden_dim)

        h = torch.tanh(h_enc + h_dec) # (batch_size, max_len, hidden_dim)
        h = self.V(h) # (batch_size, max_len, 1)
        h = h.squeeze(2) # (batch_size, max_len)
        h = torch.softmax(h, dim=1) # (batch_size, max_len)
        h = h.unsqueeze(1) # (batch_size, 1, max_len)
        h = torch.bmm(h, h_enc) # (batch_size, 1, hidden_dim)
        h = h.squeeze(1) # (batch_size, hidden_dim) # This is the context vector

        return h

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.attention = Attention(hidden_dim)
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs):
        '''
        input: (batch_size)
        hidden: (n_layers, batch_size, hidden_dim)
        encoder_outputs: (batch_size, max_len, hidden_dim)
        '''
        input = input.unsqueeze(0) # (1, batch_size)
        embedded = self.embedding(input) # (1, batch_size, hidden_dim)
        embedded = self.dropout(embedded)
        a = self.attention(encoder_outputs, hidden[-1]) # (batch_size, hidden_dim)
        a = a.unsqueeze(0) # (1, batch_size, hidden_dim)
        rnn_input = torch.cat((embedded, a), dim=2) # (1, batch_size, hidden_dim*2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc(output.squeeze(0)) # (batch_size, output_dim)

        return prediction, hidden

if __name__ == '__main__':
    h_enc = torch.rand(8, 100, 128)
    h_dec = torch.rand(8, 128)
    attention = Attention(128)
    h = attention(h_enc, h_dec)
    print(h.shape)