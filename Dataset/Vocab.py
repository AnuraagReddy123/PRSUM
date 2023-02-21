import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import json
import Constants
import os
import pickle

from collections import Counter
from nltk import word_tokenize

join = os.path.join


class Vocab(object):
    START = 0
    BLANK = 1
    END = 2
    UNK = 3

    def __init__(self) -> None:
        self.word2idx = {}
        self.word2count = Counter()
        self.reserved = ['<START>', '<BLANK>', '<END>', '<UNK>']
        self.index2word = self.reserved[:]
    
    def add_words(self, words):
        """Add a new token to the vocab and do mapping between word and index.

        Args:
            words (list): The list of tokens to be added.
        """
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)
    
    def __getitem__(self, item):
        """
        Get the index of a token or the token of an index.

        Args:
            item (str or int): The token or the index.
        """
        if type(item) is int:
            return self.index2word[item]
        return self.word2idx.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)
    
    def size(self):
        """Return the size of the vocab."""
        return len(self.index2word)

def build_vocab(data):
    """
    Build a vocab from the data.

    Args:
        data: File names of the data.
        Each file contains a Pull Request

    Returns:
        Vocab: The vocab built.
    """

    word_counts = Counter()
    for i in range(len(data)):
        pr = json.load(open(join('data', data[i].strip()+'.json'), 'r'))
        for word in word_tokenize(pr['body']):
            word_counts[word] += 1
        for word in word_tokenize(pr['issue_title']):
            word_counts[word] += 1
        
        commits = pr['commits']
        for sha in commits:
            for word in word_tokenize(commits[sha]['cm']):
                word_counts[word] += 1
            for word in word_tokenize(commits[sha]['comments']):
                word_counts[word] += 1

            graphs = commits[sha]['graphs']
            for graph in graphs:
                for node in graph['node_features']:
                    word_counts[node[0]] += 1
                    word_counts[node[1]] += 1
                    word_counts[node[2]] += 1
    
    vocab = Vocab()
    
    for word, count in word_counts.most_common(Constants.MAX_VOCAB-4):
        vocab.add_words([word])
    
    return vocab

def pr_to_index(pr, vocab:Vocab):
    '''
    Convert the source code to index
    '''
    unk_id = vocab.UNK
    def _word2idx(source):
        oovs = []
        ids = []
        
        if source == None:
            return [unk_id], oovs
        
        for w in word_tokenize(source):
            i = vocab[w]
            if i == unk_id:
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)
                ids.append(vocab.size() + oov_num)
            else:
                ids.append(i)
        return ids, oovs

    oovs = []
    
    pr['issue_title'], oov = _word2idx(pr['issue_title'])
    oovs.extend(oov)
    
    for commit_sha in pr['commits']:
        cm = pr['commits'][commit_sha]['cm']
        pr['commits'][commit_sha]['cm'], oov = _word2idx(cm)
        oovs.extend(oov)

        comments = pr['commits'][commit_sha]['comments']
        pr['commits'][commit_sha]['comments'], oov = _word2idx(comments)
        oovs.extend(oov)

        graphs = pr['commits'][commit_sha]['graphs']
        for graph in graphs:
            node_features = graph['node_features']
            for i in range(len(node_features)):
                # node_features[i][0], oov = _word2idx(node_features[i][0])
                # oovs.extend(oov)
                # node_features[i][1], oov = _word2idx(node_features[i][1])
                # oovs.extend(oov)
                # node_features[i][2], oov = _word2idx(node_features[i][2])
                # oovs.extend(oov)
                # node_features[i] = [x[0] for x in node_features[i]]
                node_features[i] = [vocab[x] for x in node_features[i]]

    oovs = list(set(oovs))
    pr['oovs'] = oovs

    body_ids = []
    for word in word_tokenize(pr['body']):
        i = vocab[word]
        if i == unk_id:
            if word in oovs:
                vocab_idx = vocab.size() + oovs.index(word)
                body_ids.append(vocab_idx)
            else:
                body_ids.append(unk_id)
        else:
            body_ids.append(i)
    
    pr['body'] = body_ids
    return pr


if __name__ == '__main__':
    fns_train = open(join('data_train.txt'), 'r').readlines()
    vocab = build_vocab(fns_train)
    pickle.dump(vocab, open('vocab.pkl', 'wb'))
    
    if not os.path.exists('data3'):
        os.mkdir('data3')

    for i in range(len(fns_train)):
        print('Processing %d/%d' % (i+1, len(fns_train)))
        pr = json.load(open(join('data', fns_train[i].strip()+'.json'), 'r'))
        pr = pr_to_index(pr, vocab)
        json.dump(pr, open(join('data3', fns_train[i].strip()+'.json'), 'w'))

