import sys
sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn as nn
from Model import Model
import Constants
from Train import generate_batch
from Loss import bleu4
from rouge import Rouge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def tensor_to_text(prdesc_tensor, vocab):

    text = ""
    for word_idx in prdesc_tensor:
        text += vocab[int(word_idx)] + " "

    return text

if __name__=='__main__':

    modelname = sys.argv[1]
    filename = sys.argv[2]

    if modelname not in ['final', 'best_train', 'best_valid']:
        print("Invalid model name")
        exit(0)
    
    if filename not in ['train', 'test', 'valid']:
        print("Invalid filename")
        exit(0)

    # n_points = int(sys.argv[1])

    # fns_test = open('../Dataset/data_train.txt').readlines()
    fns_test = open('../Dataset/data_'+filename+'.txt').readlines()

    model = Model(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBED_DIM, Constants.NUM_LAYERS).to(device)
    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('model_final.pt'))
    model.load_state_dict(torch.load('models/model_'+modelname+'.pt'))

    vocab = eval(open('../Dataset/vocab.txt').read())
    print(len(vocab))

    r = Rouge()

    bleu_total = 0.0
    rouge_1_total = 0.0
    rouge_2_total = 0.0
    rouge_l_total = 0.0

    prediction_file = open('predictions/'+modelname+'_'+filename+'.txt', 'w+')

    for num, (batch_pr, batch_prdesc, batch_prdesc_shift) in enumerate(generate_batch(fns_test, Constants.BATCH_SIZE)):

        pred_batch_prdesc = model.module.predict(batch_pr, Constants.MAX_LEN)

        for i in range(len(batch_pr)):

            prediction_file.write(f'Datapoint num: {num*Constants.BATCH_SIZE + i}\n')
            gt = tensor_to_text(batch_prdesc[i], vocab)
            pred = tensor_to_text(pred_batch_prdesc[i], vocab)
            # Take only uptill the END token
            gt1 = gt.split('<END>')[0].strip().split() + ['<END>']
            pred1 = pred.split('<END>')[0].strip().split() + ['<END>']

            bleu = bleu4(gt1, pred1)
            r_score = r.get_scores(' '.join(pred1), ' '.join(gt1))[0]

            prediction_file.write(f"Ground Truth:\n{gt}\n\nPrediction:\n{pred}\n")
            prediction_file.write(f"Bleu: {bleu}\nRouge-1: {r_score['rouge-1']['f']}\nRouge-2: {r_score['rouge-2']['f']}\nRouge-L: {r_score['rouge-l']['f']}\n\n--------------------\n\n")

            bleu_total += bleu
            rouge_1_total += r_score['rouge-1']['f']
            rouge_2_total += r_score['rouge-2']['f']
            rouge_l_total += r_score['rouge-l']['f']
    
    bleu_total /= len(fns_test)
    rouge_1_total /= len(fns_test)
    rouge_2_total /= len(fns_test)
    rouge_l_total /= len(fns_test)

    prediction_file.write(f"Total Avg Results:\n\nBleu: {bleu_total}\nRouge-1: {rouge_1_total}\nRouge-2: {rouge_2_total}\nRouge-L: {rouge_l_total}\n")