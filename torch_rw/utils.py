import torch

def edge_tensor(graph):
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    edge_index = []
    for edge in edges:
        head_index = nodes.index(edge[0])
        tail_index = nodes.index(edge[1])
        edge_index.append([head_index,tail_index])

    # to tensor
    edge_tensor = torch.LongTensor(edge_index)
    return edge_tensor


def nodes_tensor(graph):
    nodes = list(graph.nodes())
    nodes_index = []
    for node in nodes:
        nodes_index.append(nodes.index(node))

    nodes_t = torch.LongTensor(nodes_index)
    return nodes_t
