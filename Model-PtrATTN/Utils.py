import torch
import json
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os

join = os.path.join


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_graph(path_graph):
    '''
    Return dataloader from graph.json and create the Data object
    '''
    with open(path_graph) as f:
        graphs = json.load(f)

    dataset = []

    # Create the Data object, each graph is a data object
    for graph in graphs:
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(graph['edge_attr'], dtype=torch.float)
        x = torch.tensor(graph['node_features'], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
        data.validate(raise_on_error=True)
        dataset.append(data)
    
    # Create a dataloader
    return DataLoader(dataset, batch_size=2, shuffle=True)

def replace_oovs_pr(batch_pr, vocab):
    """
    Replaces the OOVs in a sequence of words with <UNK> token.
    
    Args:
        batch_pr: Batch of pull requests dictionaries (batch_size, )
        vocab: Vocabulary object
    
    Returns:
        pr: Batch of pull requests dictionaries (batch_size, )
    """

    for pr in batch_pr:
        pr['body'] = pr['body'].to(device)
        oov_token = torch.full(pr['body'].shape, vocab['<UNK>'], dtype=torch.long).to(device)
        pr['body'] = torch.where(pr['body'] > len(vocab)-1, oov_token, pr['body'])
        
        pr['issue_title'] = pr['issue_title'].to(device)
        oov_token = torch.full(pr['issue_title'].shape, vocab['<UNK>'], dtype=torch.long).to(device)
        pr['issue_title'] = torch.where(pr['issue_title'] > len(vocab)-1, oov_token, pr['issue_title'])

        for commit in pr['commits'].values():
            commit['cm'] = commit['cm'].to(device)
            oov_token = torch.full(commit['cm'].shape, vocab['<UNK>'], dtype=torch.long).to(device)
            commit['cm'] = torch.where(commit['cm'] > len(vocab)-1, oov_token, commit['cm'])

            commit['comments'] = commit['comments'].to(device)
            oov_token = torch.full(commit['comments'].shape, vocab['<UNK>'], dtype=torch.long).to(device)
            commit['comments'] = torch.where(commit['comments'] > len(vocab)-1, oov_token, commit['comments'])

            graphs = commit['graphs']
            for graph in graphs:
                graph['node_features'] = graph['node_features'].to(device)
                oov_token = torch.full(graph['node_features'].shape, vocab['<UNK>'], dtype=torch.long).to(device)
                graph['node_features'] = torch.where(graph['node_features'] > len(vocab)-1, oov_token, graph['node_features'])

    return batch_pr

def replace_oovs (in_tensor, vocab):
    """
    Replaces the OOVs in a sequence of words with <UNK> token.
    
    Args:
        in_tensor: Tensor of shape (batch_size, seq_len) containing the indices of the words in the vocabulary
        vocab: Vocabulary object
    
    Returns:
        out_tensor: Tensor of shape (batch_size, seq_len) containing the indices of the words in the vocabulary
    """
    oov_token = torch.full(in_tensor.shape, vocab['<UNK>'], dtype=torch.long).to(device)
    out_tensor = torch.where(in_tensor > len(vocab)-1, oov_token, in_tensor)
    return out_tensor

def get_pr_mask(batch_pr):
    '''
    Return a mask for the pull requests in the batch

    Args:
        batch_pr: Batch of pull requests dictionaries (batch_size, )
    
    Returns:
        pr_mask: Mask for the pull requests in the batch (batch_size, )
    '''

    pr_mask = []
    for pr in batch_pr:
        mask = []
        mask.append(torch.where(pr['issue_title'] != 0, torch.ones_like(pr['issue_title']), torch.zeros_like(pr['issue_title'])))
        for commit in pr['commits'].values():
            mask.append(torch.where(commit['cm'] != 0, torch.ones_like(commit['cm']), torch.zeros_like(commit['cm'])))
            mask.append(torch.where(commit['comments'] != 0, torch.ones_like(commit['comments']), torch.zeros_like(commit['comments'])))
        pr_mask.append(torch.cat(mask, dim=0)) # (seq_len, )
    
    return torch.stack(pr_mask, dim=0) # (batch_size, seq_len)

def extract_useful_pr(batch_pr):
    '''
    Extract the useful information from the pull requests

    Args:
        batch_pr: Batch of pull requests dictionaries (batch_size, )
    
    Returns:
        pr: Batch of pull requests dictionaries (batch_size, )
    '''

    prs = []
    for pr in batch_pr:
        mini_pr = []
        mini_pr.append(pr['issue_title'])
        for commit in pr['commits'].values():
            mini_pr.append(commit['cm'])
            mini_pr.append(commit['comments'])
        prs.append(torch.cat(mini_pr, dim=0)) # (seq_len, )

    return torch.stack(prs, dim=0) # (batch_size, seq_len)