import torch_rw_native

def walk(row_ptr,col_idx,target_nodes, p, q, walk_length):
    return torch_rw_native.walk(row_ptr,col_idx,target_nodes,p,q,walk_length)