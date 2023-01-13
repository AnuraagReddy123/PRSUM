import torch
import torch.nn as nn
import numpy as np
import math


def loss_fn (logits, target_prdesc):
    '''
    Calculate the masked loss

    Parameters:
        logits: The logits
            Shape: (batch_size, max_pr_len, vocab_size)
        target_prdesc: The target prdesc
            Shape: (batch_size, max_pr_len)
    '''
    # Mask the loss
    mask = (target_prdesc != 1).float() # The padding value is 1
    # print("Mask: ", mask)
    # Transpose the logits
    logits = logits.transpose(1, 2)
    loss = nn.CrossEntropyLoss(reduction='none')(logits, target_prdesc)
    # print(loss, loss.shape)
    loss = loss * mask
    # print(loss, loss.shape)
    # exit(0)
    loss = loss.sum() / mask.sum()

    return loss

def accuracy_fn (logits, target_prdesc):
    '''
    Calculate the masked accuracy

    Parameters:
        logits: The logits
            Shape: (batch_size, max_pr_len, vocab_size)
        target_prdesc: The target prdesc
            Shape: (batch_size, max_pr_len)
    '''
    # Mask the accuracy
    mask = (target_prdesc != 1).float() # The padding value is 1
    pred = torch.argmax(logits, dim=-1)
    correct = (pred == target_prdesc).float()
    correct = correct * mask
    accuracy = correct.sum() / mask.sum()

    # print("Pred: ", pred)
    # print("Target :", target_prdesc)
    # print("Correct: ", correct.sum(), " Mask: ", mask.sum())

    return accuracy 


def ngram(words, n):
    return list(zip(*(words[i:] for i in range(n))))

def bleu4(true, pred):
    c = len(pred)
    r = len(true)
    bp = 1. if c > r else np.exp(1 - r / (c + 1e-10))
    score = 0
    #print("True NGRAM: ", pred[0])
    for i in range(1, 5):
        true_ngram = set(ngram(true, i))
        pred_ngram = ngram(pred, i)
        length = float(len(pred_ngram)) + 1e-10    
        count = sum([1. if t in true_ngram else 0. for t in pred_ngram])
        score += math.log(1e-10 + (count / length))
    score = math.exp(score * .25)
    bleu = bp * score
    return bleu



if __name__ == '__main__':

    score = bleu4(['eclips', 'build', 'file', 'miss', 'eclips', 'project', 'file', 'gener', 'close' ], ['eclips', 'build', 'file', 'miss', 'eclips', 'eclips', 'file', 'gener'])

    # score = bleu4(['eclips', 'build', 'file', 'miss', 'eclips', 'project', 'file', 'gener', 'close' ], ['eclips', 'build', 'file', 'miss', 'eclips', 'project', 'file', 'gener', 'close' ])

    print(score)