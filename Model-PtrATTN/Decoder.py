import torch
import torch.nn as nn
import torch.nn.functional as F
import Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

class Decoder(nn.Module):
    def __init__ (self, vocab_size, hidden_dim, embed_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)

        self.W1 = nn.Linear(self.hidden_dim*3, hidden_dim)
        self.W2 = nn.Linear(self.hidden_dim, vocab_size)
        self.w_gen = nn.Linear(self.hidden_dim*4 + embed_dim, 1)
    
    def forward(self, x_t, h_dec, c_dec, context_vector):
        """Define forward propagation for the decoder.

        Args:
            x_t (Tensor):
                The input of the decoder x_t of shape (batch_size, 1).
            decoder_states (tuple):
                The hidden states(h_n, c_n) of the decoder from last time step.
                The shapes are (1, batch_size, hidden_units) for each.
            context_vector (Tensor):
                The context vector from the attention network
                of shape (batch_size,2*hidden_units).

        Returns:
            p_vocab (Tensor):
                The vocabulary distribution of shape (batch_size, vocab_size).
            docoder_states (tuple):
                The lstm states in the decoder.
                The shapes are (1, batch_size, hidden_units) for each.
            p_gen (Tensor):
                The generation probabilities of shape (batch_size, 1).
        """
        decoder_emb = self.embedding(x_t)

        decoder_output, (h_dec, c_dec) = self.lstm(decoder_emb, (h_dec, c_dec))

        # concatenate context vector and decoder state
        # (batch_size, 3*hidden_units)
        decoder_output = decoder_output.view(-1, self.hidden_dim)
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1)

        # calculate vocabulary distribution
        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)
        # (batch_size, vocab_size)
        FF2_out = self.W2(FF1_out)
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)

        # Concatenate h and c to get s_t and expand the dim of s_t.
        s_t = torch.cat([h_dec, c_dec], dim=2) # (1, batch_size, 2*hidden_units)

        p_gen = None   
        # Calculate p_gen.
        # Refer to equation (8).
        x_gen = torch.cat([context_vector, s_t.squeeze(0), decoder_emb.squeeze(1)], dim=-1)
        p_gen = torch.sigmoid(self.w_gen(x_gen))

        return p_vocab, h_dec, c_dec, p_gen

# class Decoder(nn.Module):

#     def __init__ (self, vocab_size, hidden_dim, embed_dim, num_layers):
#         super(Decoder, self).__init__()

#         self.hidden_dim = hidden_dim
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.num_layers = num_layers
#         self.atten = Attention(hidden_dim)
#         self.ptrgen = PtrGen(hidden_dim)

#         self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
#         self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True) # Just a single layer LSTM
#         self.wc = nn.Linear(2*hidden_dim, hidden_dim, bias=False)
#         self.linear = nn.Sequential(
#             nn.Linear(hidden_dim, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, vocab_size)
#         )

#     def forward (self, input, batch_enc, batch_h, batch_c):
#         '''
#         input: (batch_size, 1)
#         batch_enc: (batch_size, max_pr_len, hidden_dim)
#         batch_h: (num_layers, batch_size, hidden_dim)
#         batch_c: (num_layers, batch_size, hidden_dim)
#         '''

#         embedding = self.emb(input) # (batch_size, 1, embed_dim)
#         out, (h, c) = self.lstm(embedding, (batch_h, batch_c)) # (batch_size, 1, hidden_dim), (1, batch_size, hidden_dim), (1, batch_size, hidden_dim)

#         context, attention = self.atten((h, c), batch_enc) 












#         # Pointer Generator and Attention
#         context, attention = self.atten(batch_enc, out) # (batch_size, 1, hidden_dim) and (batch_size, max_pr_len)
#         p_gen = self.ptrgen(embedding, out, context) # (batch_size, 1, 1)

#         # Concatenate context and outt
#         out = torch.squeeze(out, 1) # (batch_size, hidden_dim)
#         out = torch.cat((context, out), 1) # (batch_size, 2*hidden_dim)

#         # Linear
#         out = self.wc(out) # (batch_size, hidden_dim)

#         # Linear
#         out = self.linear(out) # (batch_size, vocab_size)

#         # Softmax
#         out = torch.softmax(out, dim=1) # (batch_size, vocab_size)

#         # Pointer Generator
#         out = p_gen*out + (1-p_gen)*context # (batch_size, 1, hidden_dim)

#         # take log
#         out = torch.log(out+1e-10)

#         return out, h, c
    
#     def predict (self, batch_enc, batch_h, batch_c):
#         '''
#         batch_enc: (batch_size, max_pr_len, hidden_dim)
#         batch_h: (num_layers, batch_size, hidden_dim)
#         batch_c: (num_layers, batch_size, hidden_dim)
#         '''
#         batch_size = batch_enc.shape[0]
#         input = torch.zeros(batch_size, 1).long().to(device)
#         outputs = []
#         for i in range(Constants.MAX_LEN):
#             out, batch_h, batch_c = self.forward(input, batch_enc, batch_h, batch_c)
#             out = torch.argmax(out, dim=1)
#             outputs.append(out)
#             input = out.unsqueeze(1)
#         outputs = torch.stack(outputs, dim=1)
#         return outputs


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


