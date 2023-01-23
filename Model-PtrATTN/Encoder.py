import sys
sys.path.append('.')
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import Constants

from Layers import GCN

MAX_GRAPHS = Constants.N_GRAPHS
MAX_COMMITS = Constants.N_COMMITS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embed_dim, node_dim, num_layers):
        super(Encoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.node_dim = node_dim

        # !!!!!!!!!!!!!!!!!!!!
        # Separate embedding layers for graph and text. Should they be the same? Should graph have different vocabularies?
        self.embedding = nn.Embedding(vocab_size, self.embed_dim, padding_idx=1)
        self.graph_emb = nn.Embedding(vocab_size, self.embed_dim)
        
        self.enc_commit_msgs = nn.LSTM(embed_dim, hidden_dim,  num_layers=num_layers, batch_first=True, dropout=0.1)
        self.enc_src_comments = nn.LSTM(embed_dim, hidden_dim,  num_layers=num_layers, batch_first=True, dropout=0.1)
        self.enc_issue_titles = nn.LSTM(embed_dim, hidden_dim,  num_layers=num_layers, batch_first=True, dropout=0.1)

        # Number of graph layers can be adjusted
        self.gcn = GCN(node_dim*embed_dim, hidden_dim)
        self.lin_nodemerge = nn.Linear(hidden_dim, Constants.NODE_HID_DIM)

        self.lin_astmergeh = nn.Linear(Constants.NODE_HID_DIM*Constants.GRAPH_N_NODES, Constants.GRAPH_HID_DIM)
        self.lin_astmergec = nn.Linear(Constants.NODE_HID_DIM*Constants.GRAPH_N_NODES, Constants.GRAPH_HID_DIM)

        self.lin_mergeh = nn.Linear(2*hidden_dim+MAX_GRAPHS*Constants.GRAPH_HID_DIM, Constants.COMMIT_HID_DIM)
        self.lin_mergec = nn.Linear(2*hidden_dim+MAX_GRAPHS*Constants.GRAPH_HID_DIM, Constants.COMMIT_HID_DIM)

        self.lin_finmergeh = nn.Linear(MAX_COMMITS*Constants.COMMIT_HID_DIM+hidden_dim, hidden_dim)
        self.lin_finmergec = nn.Linear(MAX_COMMITS*Constants.COMMIT_HID_DIM+hidden_dim, hidden_dim)
    
    def initialize_hidden_state(self):
        return torch.zeros((self.num_layers, 1, self.hidden_dim)).to(device), torch.zeros((self.num_layers, 1, self.hidden_dim)).to(device)

    def forward(self, batch_pr):
        
        batch_h = []
        batch_c = []
        batch_enc = []

        for pr in batch_pr:
            enc, h, c = self.encode(pr)
            batch_h.append(h)
            batch_c.append(c)
            batch_enc.append(enc)

        batch_h = torch.cat(batch_h, dim=1) # (num_layers, batch_size, hidden_dim)
        batch_c = torch.cat(batch_c, dim=1) # (num_layers, batch_size, hidden_dim)
        batch_enc = torch.cat(batch_enc, dim=0) # (batch_size, seq_len, hidden_dim)

        return batch_enc, batch_h, batch_c


    def encode(self, pr):
        commits = pr['commits']

        enc_commits = []
        h_commits = []
        c_commits = []

        for commit in commits.values():
            inp_sc = commit['comments']
            inp_commit = commit['cm']

            # convert to tensor
            inp_sc = torch.tensor(inp_sc).to(device)
            inp_commit = torch.tensor(inp_commit).to(device)

            # Increase dim
            inp_sc = inp_sc.unsqueeze(0)
            inp_commit = inp_commit.unsqueeze(0)

            # Embedding
            emb_src_comments = self.embedding(inp_sc) # (1, seq_len, emb_dim)
            emb_commit_msgs = self.embedding(inp_commit) # (1, seq_len, emb_dim)

            # Encoding
            h0, c0 = self.initialize_hidden_state()
            enc_src_comments, (h_src_comments, c_src_comments) = self.enc_src_comments(emb_src_comments, (h0, c0)) # (batch_size=1, seq_len, hidden_dim), (num_layers, 1, hidden_dim), (num_layers, 1, hidden_dim)
            enc_commits.append(enc_src_comments)

            h0, c0 = self.initialize_hidden_state()
            enc_commit_msgs, (h_commit_msgs, c_commit_msgs) = self.enc_commit_msgs(emb_commit_msgs, (h0, c0)) # (1, seq_len, hidden_dim), (num_layers, 1, hidden_dim), (num_layers, 1, hidden_dim)
            enc_commits.append(enc_commit_msgs)

            # Get graphs
            graphs = commit['graphs'] # This is the diff graph


            h_graph = []
            c_graph = []
            for graph in graphs:
                # Create the Data object
                edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(graph['edge_type'], dtype=torch.long)
                x = torch.tensor(graph['node_features'], dtype=torch.long)

                x = x.to(device)
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)

                # Pass it through embedding layer
                x = self.graph_emb(x)
                x = x.reshape(x.shape[0], -1)                

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)
                
                data.validate(raise_on_error=True)

                # GCN
                data = self.gcn(data) # (num_of_nodes, hidden_dim)

                # Get the graph embedding
                # !!!!!!!!!!!!!!!!!!!!!
                # This is where the graph embedding is created. Should it be the mean of the node embeddings?
                # Linear layer won't work because the number of nodes is different for each graph
                # graph_emb = data.mean(dim=0) # (hidden_dim)
                graph_emb = self.lin_nodemerge(data) # (num_of_nodes, graph_hidden_dim)
                graph_emb = torch.flatten(graph_emb) # (num_of_nodes*graph_hidden_dim)

                # Fix number of nodes
                MAX_ALLOWED = Constants.GRAPH_N_NODES*Constants.GRAPH_HID_DIM
                if graph_emb.shape[0] < MAX_ALLOWED:
                    graph_emb = torch.cat((graph_emb, torch.zeros(MAX_ALLOWED-graph_emb.shape[0]).to(device)), dim=0)
                else:
                    graph_emb = graph_emb[:MAX_ALLOWED]

                h_graph.append(graph_emb)
                c_graph.append(graph_emb)
            
            h_graph = torch.stack(h_graph, dim=0) # (num_graphs, num_of_nodes*graph_hidden_dim)
            c_graph = torch.stack(c_graph, dim=0) # (num_graphs, num_of_nodes*graph_hidden_dim)

            # h_graph = torch.t(self.lin_astmergeh(h_graph)) # (1, num_of_graphs)
            # c_graph = torch.t(self.lin_astmergec(c_graph)) # (1, num_of_graphs)

            h_graph = self.lin_astmergeh(h_graph) # (num_of_graphs, graph_hidden_dim)
            c_graph = self.lin_astmergec(c_graph) # (num_of_graphs, graph_hidden_dim)
            h_graph = torch.reshape(h_graph, (1, -1)) # (1, num_of_graphs*graph_hidden_dim)
            c_graph = torch.reshape(c_graph, (1, -1)) # (1, num_of_graphs*graph_hidden_dim)

            h_graph = torch.stack([h_graph]*self.num_layers) # (num_layers, 1, num_of_graphs*graph_hidden_dim)
            c_graph = torch.stack([c_graph]*self.num_layers) # (num_layers, 1, num_of_graphs*graph_hidden_dim)

            # Concatenate
            h_commit = torch.cat((h_src_comments, h_commit_msgs, h_graph), dim=2) # (num_layers, batch_size=1, 2*hidden_dim+num_of_graphs*graph_hidden_dim)
            c_commit = torch.cat((c_src_comments, c_commit_msgs, c_graph), dim=2) # (num_layers, batch_size=1, 2*hidden_dim+num_of_graphs*graph_hidden_dim)

            h_commits.append(h_commit)
            c_commits.append(c_commit)

        # Make tensor
        h_commits = torch.cat(h_commits, dim=1) # (num_layers, num_commits, 2*hidden_dim+num_of_graphs*graph_hidden_dim)
        c_commits = torch.cat(c_commits, dim=1) # (num_layers, num_commits, 2*hidden_dim+num_of_graphs*graph_hidden_dim)
        enc_commits = torch.cat(enc_commits, dim=1) # (1, num_commits*comment_seq_len+num_commits*commit_seq_len, hidden_dim)

        # Merge all commits
        h_commits = self.lin_mergeh(h_commits) # (num_layers, num_commits, commit_hidden_dim)
        c_commits = self.lin_mergec(c_commits) # (num_layers, num_commits, commit_hidden_dim)

        # Flatten
        h_commits = torch.reshape(h_commits, (self.num_layers, 1, -1)) # (num_layers, 1, num_commits*commit_hidden_dim)
        c_commits = torch.reshape(c_commits, (self.num_layers, 1, -1)) # (num_layers, 1, num_commits*commit_hidden_dim)

        # Encode the issue
        inp_issue = pr['issue_title']
        inp_issue = torch.tensor(inp_issue).to(device)
        inp_issue = inp_issue.unsqueeze(0)

        emb_issue_titles = self.embedding(inp_issue) # (1, seq_len, emb_dim)
        h0, c0 = self.initialize_hidden_state()
        enc_issue_titles, (h_issue_titles, c_issue_titles) = self.enc_issue_titles(emb_issue_titles, (h0, c0)) # (1, seq_len, hidden_dim), (num_layers, 1, hidden_dim), (num_layers, 1, hidden_dim)

        # Concatenate
        h = torch.cat((h_commits, h_issue_titles), dim=2) # (num_layers, 1, num_commits*commit_hidden_dim + hidden_dim)
        c = torch.cat((c_commits, c_issue_titles), dim=2) # (num_layers, 1, num_commits*commit_hidden_dim + hidden_dim)
        enc = torch.cat((enc_commits, enc_issue_titles), dim=1) # (1, num_commits*comment_seq_len+num_commits*commit_seq_len+seq_len, hidden_dim)

        # Merge
        h = self.lin_finmergeh(h) # (num_layers, 1, hidden_dim)
        c = self.lin_finmergec(c) # (num_layers, 1, hidden_dim)

        return enc, h, c


if __name__ == '__main__':
    # Load data
    vocab_size = 100
    emb_dim = 10
    hidden_dim = 10

    batch_size = 2
    num_commits = MAX_COMMITS
    num_graphs = MAX_GRAPHS

    batch_pr = []
    for i in range(batch_size):
        pr = {}
        pr['issue_title'] = torch.randint(0, vocab_size, (10,))
        pr['commits'] = {}
        for j in range(num_commits):
            commit = {}
            commit['comments'] = torch.randint(0, vocab_size, (10,))
            commit['cm'] = torch.randint(0, vocab_size, (10,))
            commit['graphs'] = []
            for k in range(num_graphs):
                graph = {}
                graph['edge_index'] = torch.randint(0, 10, (10, 2))
                graph['edge_type'] = torch.randint(0, 3, (10, ))
                graph['node_features'] = torch.randint(0, 100, (10, 3))
                commit['graphs'].append(graph)
            pr['commits'][j] = commit
        batch_pr.append(pr)
    
    # Create model
    model = Encoder(vocab_size, emb_dim, hidden_dim, 3, num_layers=3)
    model.to(device)
    h, c = model(batch_pr)

    print(h.shape)
    print(c.shape)
    print(h)
    print(c)

    
