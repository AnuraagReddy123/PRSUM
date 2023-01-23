import torch
import torch.nn as nn
from Attention import Attention

class PtrGen(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.wt = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ws = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wx = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bptr = nn.Parameter(torch.zeros(1))
    
    def forward(self, input, hidden, context):
        '''
        input: (batch_size, 1, hidden_dim)
        hidden: (batch_size, 1, hidden_dim)
        encoder_outputs: (batch_size, max_pr_len, hidden_dim)
        '''

        o1 = self.wt(context) # (batch_size, 1, hidden_dim)
        o2 = self.ws(hidden) # (batch_size, 1, hidden_dim)
        o3 = self.wx(input) # (batch_size, 1, hidden_dim)
        
        # Take sigmoid
        p_gen = torch.sigmoid(o1 + o2 + o3 + self.bptr) # (batch_size, 1, 1)

        return p_gen
