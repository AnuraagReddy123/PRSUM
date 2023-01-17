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
        h_dec = h_dec.unsqueeze(1) # (batch_size, 1, hidden_dim)
        h_enc = self.W1(h_enc) # (batch_size, max_len, hidden_dim)
        h_dec = self.W2(h_dec) # (batch_size, 1, hidden_dim)
        h = torch.tanh(h_enc + h_dec) # (batch_size, max_len, hidden_dim)
        h = self.V(h) # (batch_size, max_len, 1)
        h = h.squeeze(2) # (batch_size, max_len)
        h = torch.softmax(h, dim=1) # (batch_size, max_len)
        h = h.unsqueeze(1) # (batch_size, 1, max_len)
        h = torch.bmm(h, h_enc) # (batch_size, 1, hidden_dim)
        h = h.squeeze(1) # (batch_size, hidden_dim)

        return h