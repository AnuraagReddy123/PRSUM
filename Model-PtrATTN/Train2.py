import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../Dataset')

import pickle
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import tqdm
from load_data2 import generate_batch

from PtrGen import PGN
import Constants
from Dataset.Vocab import Vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model:PGN, optimizer, vocab, fns_train, fns_valid):
    """Train the model, evaluate it and store it.

    Args:
        dataset (dataset.PairDataset): The training dataset.
        val_dataset (dataset.PairDataset): The evaluation dataset.
        v (vocab.Vocab): The vocabulary built from the training dataset.
        start_epoch (int, optional): The starting epoch number. Defaults to 0.
    """

    val_losses = np.inf
    for epoch in range(Constants.EPOCHS):
        batch_losses = []

        for batch, (batch_pr, batch_prdesc, batch_prdesc_shift, batch_oov, batch_oov_len) in enumerate(generate_batch(fns_train, Constants.BATCH_SIZE)):
            model.train()
            optimizer.zero_grad()
            loss = model(batch_pr, batch_prdesc, batch_prdesc_shift, batch_oov_len)
            batch_losses.append(loss.item())
            loss.backward()

            optimizer.step()

            if batch % 100 == 0:
                print(f"Epoch: {epoch+1}, Batch: {batch}, Loss: {loss.item()}")
                # print(f"Epoch: {epoch+1}, Batch: {batch}, Loss: {loss.item()}, Accuracy: {accuracy.item()}")


        epoch_loss = np.mean(batch_losses)
        
            




if __name__ == '__main__':
    fns_train = open('../Dataset/data_train.txt').readlines()
    fns_valid = open('../Dataset/data_valid.txt').readlines()

    vocab = pickle.load(open('../Dataset/vocab.pkl', 'rb'))
    
    model = PGN(vocab).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses, accuracies = train(model, optimizer, vocab, fns_train, fns_valid)
    x = 5