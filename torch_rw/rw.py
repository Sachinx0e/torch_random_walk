import torch_rw_native

def walk(edge_indices,target_nodes, p, q, walk_length):
    return torch_rw_native.walk(edge_indices,target_nodes,p,q,walk_length)