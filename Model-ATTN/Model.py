import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder

class Model(nn.Module):
    def __init__ (self, vocab_size, hidden_dim, embed_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.encoder = Encoder(vocab_size, hidden_dim, embed_dim, node_dim=3, num_layers=num_layers)
        self.decoder = Decoder(vocab_size, hidden_dim, embed_dim, num_layers=num_layers)

    def forward (self, batch_pr, batch_prdesc_shift):
        '''
        batch_pr: (batch_size, max_len)
        batch_prdesc_shift: (batch_size, max_len)
        batch_prdesc: (batch_size, max_len)
        '''
        h_enc, c_enc = self.encoder(batch_pr) # (1, batch_size, hidden_dim), (1, batch_size, hidden_dim)
        logits, h, c = self.decoder(batch_prdesc_shift, h_enc, c_enc) # (batch_size, max_len, vocab_size), (1, batch_size, hidden_dim), (1, batch_size, hidden_dim)

        return logits
    
    def predict (self, batch_pr, max_len):
        '''
        batch_pr: (batch_size, max_len)
        '''
        h_enc, c_enc = self.encoder(batch_pr) # (1, batch_size, hidden_dim), (1, batch_size, hidden_dim)
        batch_prdesc_shift = torch.zeros((len(batch_pr), 1), dtype=torch.long) # (batch_size, 1)
        h = h_enc
        c = c_enc
        batch_prdesc = []
        
        for i in range(max_len):
            logits, h, c = self.decoder(batch_prdesc_shift, h, c)
            batch_prdesc_shift = torch.argmax(logits, dim=-1) # (batch_size, 1)
            batch_prdesc.append(batch_prdesc_shift)
        
        batch_prdesc = torch.cat(batch_prdesc, dim=1) # (batch_size, max_len)
        return batch_prdesc