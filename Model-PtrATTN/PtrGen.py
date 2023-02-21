import sys
sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn as nn
from Attention import Attention
import Constants
from Encoder import Encoder
from Decoder import Decoder
from Utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PGN(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.attention = Attention(Constants.HIDDEN_DIM)
        self.encoder = Encoder(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBED_DIM, node_dim=3, num_layers=Constants.NUM_LAYERS).to(device)
        self.decoder = Decoder(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBED_DIM, Constants.NUM_LAYERS).to(device)
    
    def final_distribution(self, pr, p_gen, p_vocab, attention_weights, max_oov):
        """Calculate the final distribution for the model.

        Args:
            pr: batch of pull requests dictionaries (batch_size, )
            p_gen: (batch_size, 1)
            p_vocab: (batch_size, vocab_size)
            attention_weights: (batch_size, seq_len)
            max_oov: (Tensor or int): The maximum sequence length in the batch.

        Returns:
            final_distribution (Tensor):
            The final distribution over the extended vocabualary.
            The shape is (batch_size, )
        """
        batch_size = len(pr)
        # Clip the probabilities
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        # Get the weighted probabilities.
        # Refer to equation (9).
        p_vocab_weighted = p_gen * p_vocab
        # (batch_size, seq_len)
        attention_weighted = (1 - p_gen) * attention_weights
        # Get the extended-vocab probability distribution
        # extended_size = len(self.v) + max_oovs
        extension = torch.zeros((batch_size, max_oov)).float().to(device)
        # (batch_size, extended_vocab_size)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)

        # Add the attention weights to the corresponding vocab positions.
        # Refer to equation (9).
        final_distribution = \
            p_vocab_extended.scatter_add_(dim=1,
                                          index=pr,
                                          src=attention_weighted)

        return final_distribution

    def forward (self, pr, pr_desc, pr_desc_shift, len_oovs):
        """Define the forward propagation for the seq2seq model.

        Args:
            pr: batch of pull requests dictionaries (batch_size, )
            pr_desc: batch of pull request descriptions (batch_size, max_pr_desc_len)
            len_oovs: (Tensor): The length of the OOVs in the batch.
        Returns:
            batch_loss (Tensor): The average loss of the current batch.
        """

        pr_copy = replace_oovs_pr(pr, self.vocab)
        pr_mask = get_pr_mask(pr)
        pr_useful = extract_useful_pr(pr)

        # Encode the pull request
        encoder_outputs, h_enc, c_enc = self.encoder(pr_copy)
        # h_dec = h_enc.reshape([2, h_enc.shape[0]//2, h_enc.shape[1], h_enc.shape[2]])
        # c_dec = c_enc.reshape([2, c_enc.shape[0]//2, c_enc.shape[1], c_enc.shape[2]])
        h_dec = torch.sum(h_enc, dim=0).unsqueeze(0)
        c_dec = torch.sum(c_enc, dim=0).unsqueeze(0)

        # Coverage vector
        coverage_vector = torch.zeros(pr_mask.size()).to(device)

        # Initialize the loss
        step_losses = []

        for i in range(pr_desc.shape[1]):
            # Get the current input
            input = pr_desc_shift[:, i].unsqueeze(1) # (batch_size, 1)
            input = replace_oovs(input, self.vocab)
            context_vector, attention_weights, coverage_vector = self.attention(encoder_outputs, h_dec, c_dec, pr_mask, coverage_vector)

            p_vocab, h_dec, c_dec, p_gen = self.decoder(input, h_dec, c_dec, context_vector)

            # Calculate the final distribution
            final_distribution = self.final_distribution(pr_useful, p_gen, p_vocab, attention_weights, torch.max(len_oovs))

            x_t = torch.argmax(final_distribution, dim=1).to(device)

            target_probs = torch.gather(final_distribution, 1, pr_desc[:, i].unsqueeze(1))
            target_probs = target_probs.squeeze(1)

            mask = torch.ne(pr_desc[:, i], 1).byte().to(device)
            loss = -torch.log(target_probs + 1e-10)

            # Coverage loss
            ct_min = torch.min(attention_weights, coverage_vector)
            coverage_loss = torch.sum(ct_min, dim=1)
            loss += coverage_loss

            mask = mask.float()
            loss = loss * mask

            step_losses.append(loss)
        
        batch_loss = torch.sum(torch.stack(step_losses, 1), 1)
        # Get the non padded length of each sequence in the batch
        seq_len_mask = torch.ne(pr_desc, 1).float().to(device)
        seq_len = torch.sum(seq_len_mask, dim=1)

        batch_loss = torch.mean(batch_loss / seq_len)
        return batch_loss