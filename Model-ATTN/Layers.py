import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GATConv, GCNConv, RGCNConv
import torch.nn.functional as F

from Utils import get_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

'''
GCN will be made using pytorch geometric

Graph described by instance of torch_geometric.data.Data
    - data.x: node feature matrix with shape 
        [num_nodes, num_node_features]
    
    - data.edge_index: graph connectivity in COO format with type torch.long and shape
        [2, num_edges] 

    - data.edge_attr: edge feature matrix with shape 
        [num_edges, num_edge_features]
    
    - data.y: graph labels with shape 
        [num_graphs, num_classes]
    
    - data.pos: node position matrix with shape 
        [num_nodes, num_dimensions]

'''


class GCN(nn.Module):
    def __init__(self, c_in=3, c_out=2):
        super(GCN, self).__init__()
        self.conv1 = RGCNConv(c_in, c_out, num_relations=3)
        self.conv2 = RGCNConv(c_out, c_out, num_relations=3)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index, edge_type=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_type=edge_attr)
        
        return x


if __name__ == "__main__":
    path_graph = 'graph.json'
    loader = get_graph(path_graph)
    
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    for epoch in range(200):
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            print(out)
            break
        break
    
    
    
