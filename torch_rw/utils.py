import torch
import networkx as nx

def to_csr(graph):
    csr = nx.to_scipy_sparse_matrix(graph,format='csr')    
    row_ptr = torch.Tensor(csr.indptr)
    col_idx = torch.Tensor(csr.indices)
    return row_ptr, col_idx

def nodes_tensor(graph):
    nodes = list(graph.nodes())
    nodes_index = []
    for node in nodes:
        nodes_index.append(nodes.index(node))

    nodes_t = torch.LongTensor(nodes_index)
    return nodes_t
