import json
import os
from os import path

MAX_VOCAB = 50000


def compute_vocab():

    vocab_dict = {}

    def _add(word):
        if word == '<START>' or word == '<BLANK>' or word == '<END>' or word == '<UNK>':
            return
        if word in vocab_dict:
            vocab_dict[word] += 1
        else:
            vocab_dict[word] = 1

    filenames = open('data.txt').readlines()

    i = 0
    for filename in filenames:

        print(f"-------- datapoint {i} ----------")
        i += 1
        datapoint = json.load(open(f'data/{filename[:-1]}.json'))

        for x in datapoint['body'].split():
            _add(x)
        for x in datapoint['issue_title'].split():
            _add(x)

        for commit_sha in datapoint['commits']:
            for x in datapoint['commits'][commit_sha]['cm'].split():
                _add(x)
            for x in datapoint['commits'][commit_sha]['comments'].split():
                _add(x)
            for graph in datapoint['commits'][commit_sha]['graphs']:
                for node in graph['node_features']:
                    _add(node[0])
                    _add(node[1])
                    _add(node[2])

    # <START> -> 0, <BLANK> -> 1, <END> -> 2, <UNK> -> 3

    vocab_count = list(vocab_dict.items())
    vocab_count.sort(key=lambda k: k[1], reverse=True)
    if len(vocab_count) > MAX_VOCAB-4:
        vocab_count = vocab_count[:MAX_VOCAB-4]

    vocab = [x[0] for x in vocab_count]

    vocab = ['<START>', '<BLANK>', '<END>', '<UNK>'] + vocab

    return vocab



def encode_word_to_index():

    if not path.exists('data2'):
        os.mkdir('data2')

    vocab = eval(open('vocab.txt').read())

    def _index(word):
        if word in vocab:
            return vocab.index(word)
        else:
            return 3

    filenames = open('data.txt').readlines()

    i = 0
    for filename in filenames:

        print(f"-------- datapoint {i} ----------")
        i += 1
        datapoint = json.load(open(f'data/{filename[:-1]}.json'))

        datapoint['body'] = [_index(x) for x in datapoint['body'].split()]
        datapoint['issue_title'] = [_index(x) for x in datapoint['issue_title'].split()]

        for commit_sha in datapoint['commits']:
            cm = datapoint['commits'][commit_sha]['cm']
            datapoint['commits'][commit_sha]['cm'] = [_index(x) for x in cm.split()]

            comments = datapoint['commits'][commit_sha]['comments']
            datapoint['commits'][commit_sha]['comments'] = [_index(x) for x in comments.split()]

            for graph in datapoint['commits'][commit_sha]['graphs']:
                for node in graph['node_features']:
                    node[0], node[1], node[2] = _index(node[0]), _index(node[1]), _index(node[2])

        json.dump(datapoint, open(f'data2/{filename[:-1]}.json', 'w+'))
        
                    

if __name__=='__main__':

    vocab = compute_vocab()
    open('vocab.txt', 'w+').write(str(vocab))
    print("vocab saved.")
    encode_word_to_index()



        
