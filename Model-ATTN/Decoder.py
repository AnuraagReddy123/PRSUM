import torch
import torch.nn as nn
from Encoder import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Decoder(nn.Module):

    def __init__ (self, vocab_size, hidden_dim, embed_dim, num_layers):
        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,  dropout=0.1)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, vocab_size)
        )

    def forward (self, batch_prdesc, h_enc, c_enc):
        '''
        batch_prdesc: (batch_size, max_len)
        '''
        # Convert to tensor
        batch_prdesc = torch.tensor(batch_prdesc).to(device) # (batch_size, max_len)
        emb_prdesc = self.emb(batch_prdesc) # (batch_size, max_len, embed_dim)
        out, (h, c) = self.lstm(emb_prdesc, (h_enc, c_enc)) # (batch_size, max_len, hidden_dim), (num_layers, batch_size, hidden_dim), (num_layers, batch_size, hidden_dim)

        logits = self.linear(out) # (batch_size, max_len, vocab_size)

        # # softmax
        # probs = torch.softmax(logits, dim=-1)

        return logits, h, c


if __name__ == '__main__':
    batch_size = 32
    max_len = 100
    hidden_dim = 10
    vocab_size = 20
    embed_dim = 15

    batch_prdesc = torch.randint(0, vocab_size, (batch_size, max_len))
    h_enc = torch.zeros((1, batch_size, hidden_dim))
    c_enc = torch.zeros((1, batch_size, hidden_dim))

    decoder = Decoder(hidden_dim, vocab_size, embed_dim)
    logits, h, c = decoder(batch_prdesc, h_enc, c_enc)

    print(logits.shape)
    print(h.shape)
    print(c.shape)


