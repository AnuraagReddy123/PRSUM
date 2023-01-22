import torch
import torch.nn as nn
from Encoder import Encoder
from Attention import Attention
import Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Decoder(nn.Module):

    def __init__ (self, vocab_size, hidden_dim, embed_dim, num_layers):
        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.attention = Attention(hidden_dim)

        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,  dropout=0.1)
        self.wc = nn.Linear(2*hidden_dim, hidden_dim, bias=False)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, vocab_size)
        )

    def forward (self, input, batch_enc, batch_h, batch_c):
        '''
        input: (batch_size, 1)
        batch_enc: (batch_size, max_pr_len, hidden_dim)
        batch_h: (num_layers, batch_size, hidden_dim)
        batch_c: (num_layers, batch_size, hidden_dim)
        '''

        embedding = self.emb(input) # (batch_size, 1, embed_dim)
        out, (h, c) = self.lstm(embedding, (batch_h, batch_c)) # (batch_size, 1, hidden_dim), (num_layers, batch_size, hidden_dim), (num_layers, batch_size, hidden_dim)

        # Attention
        context = self.attention(batch_enc, out) # (batch_size, 1, hidden_dim)

        # Concatenate context and out
        out = torch.squeeze(out, 1) # (batch_size, hidden_dim)
        out = torch.cat((context, out), 1) # (batch_size, 2*hidden_dim)

        # Linear
        out = self.wc(out) # (batch_size, hidden_dim)

        # Linear
        out = self.linear(out) # (batch_size, vocab_size)

        return out, h, c
    
    def predict (self, batch_enc, batch_h, batch_c):
        '''
        batch_enc: (batch_size, max_pr_len, hidden_dim)
        batch_h: (num_layers, batch_size, hidden_dim)
        batch_c: (num_layers, batch_size, hidden_dim)
        '''
        batch_size = batch_enc.shape[0]
        input = torch.zeros(batch_size, 1).long().to(device)
        outputs = []
        for i in range(Constants.MAX_LEN):
            out, batch_h, batch_c = self.forward(input, batch_enc, batch_h, batch_c)
            out = torch.argmax(out, dim=1)
            outputs.append(out)
            input = out.unsqueeze(1)
        outputs = torch.stack(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    vocab_size = 100
    hidden_dim = 256
    embed_dim = 256
    num_layers = 2
    batch_size = 32
    max_pr_len = 100
    max_prdesc_len = 20
    decoder = Decoder(vocab_size, hidden_dim, embed_dim, num_layers)
    decoder = decoder.to(device)
    input_pr = torch.randint(0, vocab_size, (batch_size, max_pr_len)).to(device)
    target_prdesc_shift = torch.randint(0, vocab_size, (batch_size, max_prdesc_len)).to(device)
    target_prdesc = torch.randint(0, vocab_size, (batch_size, max_prdesc_len)).to(device)
    encoder = Encoder(vocab_size, hidden_dim, embed_dim, num_layers)
    encoder = encoder.to(device)
    h_enc, c_enc = encoder(input_pr)
    logits, h, c = decoder(target_prdesc_shift, h_enc, c_enc)
    print(logits.shape)
    print(h.shape)
    print(c.shape)


