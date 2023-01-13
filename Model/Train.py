import sys
sys.path.append('.')
sys.path.append('..')

from Model import Model
from Loss import loss_fn, accuracy_fn
import Constants

# import tensorflow as tf
import torch
import torch.nn as nn
import os
import time
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from load_data import generate_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def plotter(values, file_name):

    if file_name == 'accuracies':
        plt.ylim([0, 1])
    else:
        plt.ylim([0, 20])

    plt.plot(values)
    plt.savefig(f'{file_name}.png')

    open(f'{file_name}.txt', 'w+').write(str(values))


# @tf.function
def train_step(input_pr, target_prdesc_shift, target_prdesc, model: Model, optimizer):
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

    logits = model(input_pr, target_prdesc_shift)
    # logits = logits[:, :-1]
    target_prdesc = torch.tensor(target_prdesc, dtype=torch.long, device=device)
    loss = loss_fn(logits, target_prdesc)
    accuracy = accuracy_fn(logits, target_prdesc)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, accuracy

def valid_step(input_pr, target_prdesc_shift, target_prdesc, model: Model):

    model.eval()

    with torch.no_grad():

        logits = model(input_pr, target_prdesc_shift)
        # logits = logits[:, :-1]
        target_prdesc = torch.tensor(target_prdesc, dtype=torch.long, device=device)
        loss = loss_fn(logits, target_prdesc)
        accuracy = accuracy_fn(logits, target_prdesc)

        return loss, accuracy


def main_train(model: Model, fns_train, fns_valid, optimizer, epochs):
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
        for batch, (batch_pr, batch_prdesc_shift, batch_prdesc) in enumerate(generate_batch(fns_train, Constants.BATCH_SIZE)):

            if batch > 0:
                continue
            # print(f"batch: {batch}")

            # Train the batch
            train_loss, train_accuracy = train_step(batch_pr, batch_prdesc_shift, batch_prdesc, model, optimizer)
            train_losses.append(train_loss.item())
            train_accuracies.append(train_accuracy.item())
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.item(), train_accuracy.item()))

            if train_accuracy.item() > max_accuracy_train:
                max_accuracy_train = train_accuracy.item()
                torch.save(model.state_dict(), os.path.join('model_best_train.pt'))
                print("Model Train saved.")

        valid_acc_sum = 0.0
        num_valid_batches = math.ceil(len(fns_valid)/Constants.BATCH_SIZE)
        # validate the model
        for batch, (batch_pr, batch_prdesc_shift, batch_prdesc) in enumerate(generate_batch(fns_valid, Constants.BATCH_SIZE)):

            valid_loss, valid_accuracy = valid_step(batch_pr, batch_prdesc_shift, batch_prdesc, model)
            valid_losses.append(valid_loss.item())
            valid_accuracies.append(valid_accuracy.item())
            print('Validation: Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, valid_loss.item(), valid_accuracy.item()))

            valid_acc_sum += valid_accuracy.item()

            # print(valid_accuracy.item(), max_accuracy_valid, valid_accuracy.item() > max_accuracy_valid)

        avg_valid_acc = valid_acc_sum/num_valid_batches
        if avg_valid_acc > max_accuracy_valid:
        #if valid_accuracy.item() > max_accuracy_valid:
            max_accuracy_valid = avg_valid_acc
            torch.save(model.state_dict(), os.path.join('model_best_valid.pt'))
            print("Model Valid saved.")


        print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - start))

    return train_losses, train_accuracies

if __name__ == '__main__':

    fns_train = open('../Dataset/data_train.txt').readlines()
    fns_valid = open('../Dataset/data_valid.txt').readlines()

    model = Model(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBED_DIM, Constants.NUM_LAYERS).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("created model.")

    losses, accuracies = main_train(model, fns_train, fns_valid, optimizer, epochs=Constants.EPOCHS)

    plotter(losses, 'losses')
    plotter(accuracies, 'accuracies')

    # Save model
    torch.save(model.state_dict(), os.path.join('model_final.pt'))

    print('Done')
