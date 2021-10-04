import torch_rw_native

def walk(row_ptr,col_idx,target_nodes, p, q, walk_length,seed):
    return torch_rw_native.walk(row_ptr,col_idx,target_nodes,p,q,walk_length,seed)

def walk_edge_list(edge_list_indexed, node_edges_idx,target_nodes, p, q, walk_length, seed):
    return torch_rw_native.walk_edge_list(edge_list_indexed,target_nodes,p,q,walk_length,seed) 

def to_windows(walks, window_size, num_nodes,seed):
    return torch_rw_native.to_windows(walks, window_size, num_nodes,seed)

