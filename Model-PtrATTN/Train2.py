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

    train_losses = np.inf
    valid_losses = np.inf
    for epoch in range(Constants.EPOCHS):
        batch_losses = []

        print('Training...')
        model.train()
        for batch, (batch_pr, batch_prdesc, batch_prdesc_shift, batch_oov, batch_oov_len) in enumerate(generate_batch(fns_train, Constants.BATCH_SIZE)):
            optimizer.zero_grad()
            loss = model(batch_pr, batch_prdesc, batch_prdesc_shift, batch_oov_len)
            batch_losses.append(loss.item())
            loss.backward()

            optimizer.step()

            if batch % 10 == 0:
                print(f"Epoch: {epoch+1}, Batch: {batch}, Loss: {loss.item()}")
                # print(f"Epoch: {epoch+1}, Batch: {batch}, Loss: {loss.item()}, Accuracy: {accuracy.item()}")
        
        epoch_loss = np.mean(batch_losses)
        if epoch_loss < train_losses:
            train_losses = epoch_loss
            torch.save(model.state_dict(), os.path.join('models', 'model_best_train.pt'))
            print("Model Train saved.")

        print('Evaluating...')
        model.eval()
        with torch.no_grad():
            for batch, (batch_pr, batch_prdesc, batch_prdesc_shift, batch_oov, batch_oov_len) in enumerate(generate_batch(fns_valid, Constants.BATCH_SIZE)):
                loss = model(batch_pr, batch_prdesc, batch_prdesc_shift, batch_oov_len)
                batch_losses.append(loss.item())

                if batch % 10 == 0:
                    print(f"Epoch: {epoch+1}, Batch: {batch}, Loss: {loss.item()}")
                    # print(f"Epoch: {epoch+1}, Batch: {batch}, Loss: {loss.item()}, Accuracy: {accuracy.item()}")

        epoch_loss = np.mean(batch_losses)
        if epoch_loss < valid_losses:
            valid_losses = epoch_loss
            torch.save(model.state_dict(), os.path.join('models', 'model_best_valid.pt'))
            print("Model Valid saved.")


if __name__ == '__main__':
    fns_train = open('../Dataset/data_train.txt').readlines()
    fns_valid = open('../Dataset/data_valid.txt').readlines()

    vocab = pickle.load(open('../Dataset/vocab.pkl', 'rb'))
    
    model = PGN(vocab).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, optimizer, vocab, fns_train, fns_valid)

    # Save the model
    torch.save(model.state_dict(), os.path.join('models', 'model_final  .pt'))

    