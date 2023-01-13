import sys
sys.path.append('.')
sys.path.append('..')

import os
import json
import numpy as np
import copy
import torch
import Constants
from Model import Model

join = os.path.join

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_GRAPHS = Constants.N_GRAPHS
N_COMMITS = Constants.N_COMMITS
N_PRDESC = Constants.MAX_LEN

# write code for generating batches

default_graph =   {
    "node_features": [[1, 1, 1], [1, 1, 1]],
    "edge_type": [0],
    "edge_index": [[0, 1]]
  }

default_commit =  {
    'cm': [1],
    'comments': [1],
    'graphs': [default_graph]*N_GRAPHS
}

def pad_body(body: list):
    
    '''Fixes the size of body'''
    if len(body) >= N_PRDESC:
        body = body[:N_PRDESC-1] + [2]
    elif len(body) < N_PRDESC:
        body.append(2)
        body.extend([1]*(N_PRDESC - len(body)))

    return body

def pad_commits(commits: dict):

    n = len(commits)

    if n < N_COMMITS:
        for i in range(1, N_COMMITS-n+1):
            commits[f'key{i}'] = copy.deepcopy(default_commit)
    elif n > N_COMMITS:
        keys = list(commits.keys())
        keys = keys[N_COMMITS:]
        for k in keys:
            del commits[k]

    return commits

def pad_graphs (graphs: list):
    n = len(graphs)

    if n < N_GRAPHS:
        for i in range(1, N_GRAPHS-n+1):
            graphs.append(copy.deepcopy(default_graph))
    elif n > N_GRAPHS:
        graphs = graphs[:N_GRAPHS]

    return graphs

# Generator function will yield batches
def generate_batch(filenames, batch_size):
    # read the Dataset/data.txt -> this contains the filename
    # take "batch_size" PRs at once and process them
    # processing
        # padding
        # convert to tensors
        # make a batch
        # return it -> yield() function
    
    for i in range(0, len(filenames), batch_size):
        batch_pr = []
        batch_prdesc = []
        batch_prdesc_shift = []

        for j in range(min(batch_size, len(filenames)-i)):
            name = filenames[i+j].strip()
            pr = json.load(open(join('..', 'Dataset', 'data2', name+'.json'), 'r'))

            # Processing
            pr['body'] = torch.tensor(pad_body(pr['body'])).type(torch.long).to(device)
            pr['issue_title'] = torch.tensor(pr['issue_title'] if len(pr['issue_title']) > 0 else [1]).type(torch.long).to(device).to(device)

            commits = pr['commits']
            commits = pad_commits(commits)

            for sha in commits:
                commits[sha]['cm'] = torch.tensor(commits[sha]['cm'] if len(commits[sha]['cm']) > 0 else [1]).type(torch.long).to(device)
                commits[sha]['comments'] = torch.tensor(commits[sha]['comments'] if len(commits[sha]['comments']) > 0 else [1]).type(torch.long).to(device)

                graphs = commits[sha]['graphs']
                graphs = pad_graphs(graphs)

                for graph in graphs:
                    graph['node_features'] = torch.tensor(graph['node_features'], dtype=torch.float).to(device)
                    graph['edge_index'] = torch.tensor(graph['edge_index'], dtype=torch.long).to(device)
                    graph['edge_type'] = torch.tensor(graph['edge_type'], dtype=torch.long).to(device)
                
                commits[sha]['graphs'] = graphs

            pr['commits'] = commits

            batch_pr.append(pr)
            batch_prdesc.append(pr['body'])
            batch_prdesc_shift.append(torch.cat([torch.tensor([0]).type(torch.long).to(device), pr['body'][:-1]], dim=0))
                            
        batch_prdesc = torch.stack(batch_prdesc, dim=0)
        batch_prdesc_shift = torch.stack(batch_prdesc_shift, dim=0)

        yield batch_pr, batch_prdesc, batch_prdesc_shift

if __name__ == '__main__':
    print('Testing the generator function')
    filenames = open('../Dataset/data.txt').readlines()
    model = Model(Constants.VOCAB_SIZE, Constants.HIDDEN_DIM, Constants.EMBED_DIM, Constants.NUM_LAYERS).to(device)
    for i, batch in enumerate(generate_batch(filenames, 2)):
        print('Batch: ', i)
        batch_pr, batch_prdesc, batch_prdesc_shift = batch

        out = model(batch_pr, batch_prdesc_shift)
        print(out.shape)
        print(batch_prdesc)
        print(batch_prdesc_shift)

        if i == 2:
            break
