import sys
import os
join = os.path.join

modelname = sys.argv[1]
dataname = sys.argv[2]

'''
Usage:
    python ScoreCalculator.py modelname dataname
    modelname: ['besttrain', 'bestvalid', 'final']
    dataname: ['test', 'valid', 'train']
'''


with open(join('predictions', modelname+'_'+dataname+'.txt'), 'r', encoding='utf-8') as f:
    bleu = 0.0
    rouge1 = 0.0
    rouge2 = 0.0
    rougel = 0.0
    cntb = 0
    cntr1 = 0
    cntr2 = 0
    cntrl = 0

    for line in f:
        if line.startswith('Bleu'):
            cntb += 1
            bleu += float(line.split()[-1])
        elif line.startswith('Rouge-1'):
            cntr1 += 1
            rouge1 += float(line.split()[-1])
        elif line.startswith('Rouge-2'):
            cntr2 += 1
            rouge2 += float(line.split()[-1])
        elif line.startswith('Rouge-L'):
            cntrl += 1
            rougel += float(line.split()[-1])
    
    print('BLEU: ', bleu/cntb)
    print('ROUGE-1: ', rouge1/cntr1)
    print('ROUGE-2: ', rouge2/cntr2)
    print('ROUGE-L: ', rougel/cntrl)
