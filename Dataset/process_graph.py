'''
Convert the current graph format into the required format:
{
    "node_features":[],
    "edge_index":[],
    "edge_type": []
}
edge types -> [normal, action, seq]
'''
import json

edge_type_map = {
    'normal': 0,
    'action': 1,
    'seq': 2
}

if __name__ == "__main__":

    old_graph:dict = json.load(open('graph.json', 'r'))
    new_graph = {
        'node_features':[],
        'edge_index': [],
        'edge_type': []
    }

    for node in old_graph.values():

        node_features = [node['nodeType'], node['typeLabel'], node['label']]
        new_graph['node_features'].append(node_features)

        src = int(node['idx'])

        for link_i in range(len(node['links'])):
            dest = node['links'][link_i]
            edge_type = edge_type_map[node['link_type'][link_i]]
            new_graph['edge_index'].append([src, dest])
            new_graph['edge_type'].append(edge_type)

    json.dump(new_graph, open('graph_processed.json', 'w+'))

