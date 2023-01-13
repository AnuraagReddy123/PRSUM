import torch
import json
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


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