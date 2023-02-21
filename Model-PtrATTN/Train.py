import sys
sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn as nn
import os
import time
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
from io import StringIO
import smtplib

from Encoder import Encoder
from Decoder import Decoder
from PtrGen import PGN
import Constants 
from Loss import loss_fn, accuracy_fn
from load_data import generate_batch
import pickle
from Dataset.Vocab import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def send_email_crash_notification(crash_message):
    email = Constants.EMAIL
    send_to_mail = Constants.EMAIL
    subject = "Application crashed"

    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = send_to_mail
    msg['Subject'] = subject
    message = crash_message
    msg.attach(MIMEText(message, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(email, Constants.PASSWORD)


def plotter(values, file_name):

    if file_name == 'accuracies':
        plt.ylim([0, 1])
    else:
        plt.ylim([0, 20])

    plt.plot(values)
    plt.savefig(f'results/{file_name}.png')

    open(f'results/{file_name}.txt', 'w+').write(str(values))


def train_step(model:PGN, input_pr, target_prdesc_shift, target_prdesc, optimizer):
    '''
    Train a batch and return loss

    Parameters:
        input_pr: The input pr
            Shape: [dict] * batch_size
        target_prdesc_shift: The shifted target prdesc
            Shape: (batch_size, max_pr_len)
        target_prdesc: The target prdesc
            Shape: (batch_size, max_pr_len)
        encoder: The encoder
        decoder: The decoder
        optimizer: The optimizer
    '''
    # print("in train step")
    model.train()
    optimizer.zero_grad()

    loss = model(input_pr, target_prdesc, target_prdesc_shift)

    accuracy = accuracy_fn(logits, target_prdesc)

    loss.backward()
    optimizer.step()

    return loss.item()/target_prdesc.shape[1], accuracy.item()

def valid_step(encoder:Encoder, decoder:Decoder, input_pr, target_prdesc_shift, target_prdesc):
    '''
    Validate a batch and return loss

    Parameters:
        input_pr: The input pr
            Shape: [dict] * batch_size
        target_prdesc_shift: The shifted target prdesc
            Shape: (batch_size, max_pr_len)
        target_prdesc: The target prdesc
            Shape: (batch_size, max_pr_len)
        encoder: The encoder
        decoder: The decoder
    '''
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        batch_enc, batch_h, batch_c = encoder(input_pr)

        loss = 0
        logits = None
        dec_h, dec_c = batch_h, batch_c
        for i in range(target_prdesc.shape[1]):
            decoder_in = torch.unsqueeze(target_prdesc_shift[:, i], 1) # (batch_size, 1)
            logit, dec_h, dec_c = decoder(decoder_in, batch_enc, dec_h, dec_c)
            target_prdesc = torch.tensor(target_prdesc, dtype=torch.long, device=device)
            loss += loss_fn(logit, target_prdesc[:, i])

            logit = torch.unsqueeze(logit, 1)
            if logits is None:
                logits = logit
            else:
                logits = torch.cat((logits, logit), 1)

        accuracy = accuracy_fn(logits, target_prdesc)

    return loss.item()/target_prdesc.shape[1], accuracy.item()

def main_train(model:PGN, fns_train, fns_valid, optimizer, epochs):
    '''
    Train the model

    Parameters:
        encoder: The encoder
        decoder: The decoder
        dataset: The dataset
        optimizer: The optimizer
        epochs: The number of epochs
    '''
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    max_accuracy_train = - math.inf
    max_accuracy_valid = - math.inf

    for epoch in range(epochs):
        # print(f"epoch: {epoch+1}")
        # Get start time
        start = time.time()
        # For every batch
        for batch, (batch_pr, batch_prdesc, batch_prdesc_shift) in enumerate(generate_batch(fns_train, Constants.BATCH_SIZE)):

            # if batch > 0:
            #     continue
            # print(f"batch: {batch}")

            # Train the batch
            train_loss, train_accuracy = train_step(model, batch_pr, batch_prdesc_shift, batch_prdesc, optimizer)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss, train_accuracy))

            if train_accuracy > max_accuracy_train:
                max_accuracy_train = train_accuracy
                torch.save(model.state_dict(), os.path.join('models', 'model_best_train.pt'))
                print("Model Train saved.")

        valid_acc_sum = 0.0
        num_valid_batches = math.ceil(len(fns_valid)/Constants.BATCH_SIZE)
        # validate the model
        for batch, (batch_pr, batch_prdesc, batch_prdesc_shift) in enumerate(generate_batch(fns_valid, Constants.BATCH_SIZE)):

            valid_loss, valid_accuracy = valid_step(model, batch_pr, batch_prdesc_shift, batch_prdesc)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)
            print('Validation: Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, valid_loss, valid_accuracy))

            valid_acc_sum += valid_accuracy

            # print(valid_accuracy.item(), max_accuracy_valid, valid_accuracy.item() > max_accuracy_valid)

        avg_valid_acc = valid_acc_sum/num_valid_batches
        if avg_valid_acc > max_accuracy_valid:
        #if valid_accuracy.item() > max_accuracy_valid:
            max_accuracy_valid = avg_valid_acc
            torch.save(model.state_dict(), os.path.join('models','model_best_valid.pt'))
            print("Model Valid saved.")


        print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - start))

    return train_losses, train_accuracies

if __name__ == '__main__':

    fns_train = open('../Dataset/data_train.txt').readlines()
    fns_valid = open('../Dataset/data_valid.txt').readlines()

    # encoder = Encoder(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBED_DIM, node_dim=3, num_layers=Constants.NUM_LAYERS).to(device)
    # decoder = Decoder(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBED_DIM, num_layers=Constants.NUM_LAYERS).to(device)

    # encoder = nn.DataParallel(encoder)
    # decoder = nn.DataParallel(decoder)
    
    print("created model.")

    vocab = pickle.load(open('../Dataset/vocab.pkl', 'rb'))
    
    model = PGN(vocab)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    losses, accuracies = main_train(model, fns_train, fns_valid, optimizer, epochs=Constants.EPOCHS)

    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), os.path.join('models', 'model_final.pt'))

    plotter(losses, 'losses')
    plotter(accuracies, 'accuracies')

    print('Done')