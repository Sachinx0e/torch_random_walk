import torch_rw_native

def walk(row_ptr,col_idx,target_nodes, p, q, walk_length,seed):
    return torch_rw_native.walk(row_ptr,col_idx,target_nodes,p,q,walk_length,seed)

def walk_edge_list(edge_list_indexed, node_edge_index,target_nodes, p, q, walk_length, seed,padding_idx,restart=True):
    return torch_rw_native.walk_edge_list(edge_list_indexed,
                                          node_edge_index,
                                          target_nodes,
                                          p,
                                          q,
                                          walk_length,
                                          seed,
                                          padding_idx,
                                          restart
                                        )

def walk_triples(triples_indexed, relation_tail_index,target_nodes, walk_length,padding_idx,seed,restart=True):
    return torch_rw_native.walk_triples(triples_indexed,
                                          relation_tail_index,
                                          target_nodes,
                                          walk_length,
                                          padding_idx,
                                          restart,
                                          seed
                                        )                            


def to_windows(walks, window_size, num_nodes,seed):
    return torch_rw_native.to_windows(walks, window_size, num_nodes,seed)

def to_windows_triples(walks, window_size, num_nodes,padding_idx,triples,seed):
    return torch_rw_native.to_windows_triples(walks, window_size,num_nodes,padding_idx,triples,seed)

