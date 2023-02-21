import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Define feed-forward layers
        self.Wh = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)
        self.Ws = nn.Linear(2*hidden_dim, 2*hidden_dim)
        # wc for coverage
        self.wc = nn.Linear(1, 2*hidden_dim, bias=False)
        self.v = nn.Linear(2*hidden_dim, 1, bias=False)
    
    def forward(self, encoder_output, h_dec, c_dec, x_padding_masks, coverage_vector):
        """Define forward propagation for the attention network.

        Args:
            decoder_states (tuple):
                The hidden states from lstm (h_n, c_n) in the decoder,
                each with shape (1, batch_size, hidden_units)
            encoder_output (Tensor):
                The output from the lstm in the decoder with
                shape (batch_size, seq_len, hidden_units).
            x_padding_masks (Tensor):
                The padding masks for the input sequences
                with shape (batch_size, seq_len).
            coverage_vector (Tensor):
                The coverage vector from last time step.
                with shape (batch_size, seq_len).

        Returns:
            context_vector (Tensor):
                Dot products of attention weights and encoder hidden states.
                The shape is (batch_size, 2*hidden_units).
            attention_weights (Tensor): The shape is (batch_size, seq_length).
            coverage_vector (Tensor): The shape is (batch_size, seq_length).
        """
        # Concatenate h and c to get s_t and expand the dim of s_t.
        s_t = torch.cat([h_dec, c_dec], dim=2) # (1, batch_size, 2*hidden_units)
        s_t = s_t.transpose(0, 1) # (batch_size, 1, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output) # (batch_size, seq_length, 2*hidden_units)

        # calculate attention scores
        # Equation(11).
        # Wh h_* (batch_size, seq_length, 2*hidden_units)
        encoder_features = self.Wh(encoder_output)
        # Ws s_t (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)
        # (batch_size, seq_length, 2*hidden_units)
        att_inputs = encoder_features + decoder_features

        # Add coverage feature.
        coverage_features = self.wc(coverage_vector.unsqueeze(2))  # wc c
        att_inputs = att_inputs + coverage_features

        score = self.v(torch.tanh(att_inputs)) # (batch_size, seq_length, 1)
        attention_weights = F.softmax(score, dim=1).squeeze(2) # (batch_size, seq_length)
        attention_weights = attention_weights * x_padding_masks
        
        # Normalize attention weights after excluding padded positions.
        normalization_factor = attention_weights.sum(1, keepdim=True)
        attention_weights = attention_weights / normalization_factor
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output) # (batch_size, 1, 2*hidden_units)
        context_vector = context_vector.squeeze(1) # (batch_size, 2*hidden_units)

        # Update coverage vector.
        coverage_vector = coverage_vector + attention_weights

        return context_vector, attention_weights, coverage_vector



# class Attention(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         # self.W1 = nn.Linear(hidden_dim, hidden_dim)
#         # self.W2 = nn.Linear(hidden_dim, hidden_dim)
#         # self.V = nn.Linear(hidden_dim, 1)

#         self.Wh = nn.Linear(2*hidden_dim, 2*hidden_dim, bias=False)
#         self.Ws = nn.Linear(2*hidden_dim, 2*hidden_dim)

#         self.v = nn.Linear(2*hidden_dim, 1, bias=False)

#     def forward(self, encoder_output, decoder_states):
#         '''
#         encoder_output: (batch_size, max_len, hidden_dim)
#         decoder_states: ((num_layers, batch_size, hidden_dim), (num_layers, batch_size, hidden_dim))
#         '''

#         h_dec, c_dec = decoder_states
#         s_t = torch.cat((h_dec, c_dec), 2) # (1, batch_size, 2*hidden_dim)
#         s_t = s_t.transpose(0, 1) # (batch_size, 1, 2*hidden_dim)
#         s_t = s_t.expand_as(encoder_output).contiguous() # (batch_size, max_len, 2*hidden_dim)

#         # calculate attention scores
#         # Equation(11).
#         # Wh h_* (batch_size, seq_length, 2*hidden_units)
#         encoder_features = self.W1(encoder_output)
#         decoder_features = self.W2(s_t)
#         att_inputs = encoder_features + decoder_features # (batch_size, max_len, 2*hidden_dim)

#         score = self.v(torch.tanh(att_inputs)) # (batch_size, max_len, 1)
#         attention_weights = torch.softmax(score, dim=1).squeeze(2) # (batch_size, max_len)
#         # TODO: Add for padding

#         context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output) # (batch_size, 1, hidden_dim)
#         context_vector = context_vector.squeeze(1) # (batch_size, hidden_dim)

#         return context_vector, attention_weights





#         # # print("in attention")
#         # # h_dec = h_dec.unsqueeze(1) # (batch_size, 1, hidden_dim)
#         # h_enc = self.W1(h_enc) # (batch_size, max_len, hidden_dim)
#         # h_dec = self.W2(h_dec) # (batch_size, 1, hidden_dim)

#         # attention = None
#         # h = torch.tanh(h_enc + h_dec) # (batch_size, max_len, hidden_dim)
#         # h = self.V(h) # (batch_size, max_len, 1)
#         # h = h.squeeze(2) # (batch_size, max_len)
#         # h = torch.softmax(h, dim=1) # (batch_size, max_len)
#         # attention = h
#         # h = h.unsqueeze(1) # (batch_size, 1, max_len)
#         # h = torch.bmm(h, h_enc) # (batch_size, 1, hidden_dim)
#         # h = h.squeeze(1) # (batch_size, hidden_dim) # This is the context vector

#         # return h, attention

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